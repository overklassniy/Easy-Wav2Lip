import argparse
import configparser
import math
import os
import pickle
import subprocess
import warnings
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image
from batch_face import RetinaFace
from tqdm import tqdm

import utils.audio as audio
from enhance import upscale, load_sr
from utils.basic import load_model, g_colab
from utils.logger import get_logger

# Initialize logger
logger = get_logger()

# Suppress specific warnings to reduce log noise.
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

# Determine device and GPU id.
device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
gpu_id: int = 0 if torch.cuda.is_available() else -1

if device == "cpu":
    logger.warning("No GPU detected; inference will be performed on CPU, which is VERY SLOW!")

# Set up argument parser.
parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="Name of saved checkpoint to load weights from")
parser.add_argument("--segmentation_path", type=str, default="checkpoints/face_segmentation.pth",
                    help="Name of saved checkpoint of segmentation network", required=False)
parser.add_argument("--face", type=str, required=True,
                    help="Filepath of video/image that contains faces to use")
parser.add_argument("--audio", type=str, required=True,
                    help="Filepath of video/audio file to use as raw audio source")
parser.add_argument("--outfile", type=str, default="results/result_voice.mp4",
                    help="Video path to save result. See default for an e.g.")
parser.add_argument("--static", type=bool, default=False,
                    help="If True, then use only first video frame for inference")
parser.add_argument("--fps", type=float, default=25.0,
                    help="Can be specified only if input is a static image (default: 25)", required=False)
parser.add_argument("--pads", nargs="+", type=int, default=[0, 10, 0, 0],
                    help="Padding (top, bottom, left, right). Please adjust to include chin at least")
parser.add_argument("--wav2lip_batch_size", type=int, default=1,
                    help="Batch size for Wav2Lip model(s)")
parser.add_argument("--out_height", type=int, default=480,
                    help="Output video height. Best results are obtained at 480 or 720")
parser.add_argument("--crop", nargs="+", type=int, default=[0, -1, 0, -1],
                    help="Crop video to a smaller region (top, bottom, left, right). "
                         "If -1, value will be auto-inferred based on height/width")
parser.add_argument("--box", nargs="+", type=int, default=[-1, -1, -1, -1],
                    help="Specify a constant bounding box for the face. Syntax: top, bottom, left, right")
parser.add_argument("--rotate", action="store_true", default=False,
                    help="If true, rotates video by 90deg clockwise (for mobile videos)")
parser.add_argument("--nosmooth", type=str, default=False,
                    help="Prevent smoothing face detections over a short temporal window")
parser.add_argument("--no_seg", action="store_true", default=False,
                    help="Prevent using face segmentation")
parser.add_argument("--no_sr", action="store_true", default=False,
                    help="Prevent using super resolution")
parser.add_argument("--sr_model", type=str, default="gfpgan",
                    help="Name of upscaler - gfpgan or RestoreFormer", required=False)
parser.add_argument("--fullres", type=int, default=3,
                    help="Used only to determine if full res is used so that no resizing is done if so")
parser.add_argument("--debug_mask", type=str, default=False,
                    help="Makes background grayscale to see the mask better")
parser.add_argument("--preview_settings", type=str, default=False,
                    help="Processes only one frame")
parser.add_argument("--mouth_tracking", type=str, default=False,
                    help="Tracks the mouth in every frame for the mask")
parser.add_argument("--mask_dilation", type=float, default=150,
                    help="Size of mask dilation around mouth", required=False)
parser.add_argument("--mask_feathering", type=int, default=151,
                    help="Amount of feathering of mask around mouth", required=False)
parser.add_argument("--quality", type=str, default="Fast",
                    help="Choose between Fast, Improved and Enhanced")

# Load face predictor and mouth detector.
with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)
with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

# Global variables to store mask and bounding box information.
kernel = last_mask = x = y = w = h = None

# Determine if running in Google Colab.
g_colab_flag: bool = g_colab()
if not g_colab_flag:
    # Load configuration file.
    config = configparser.ConfigParser()
    config.read("config.ini")
    preview_window: str = config.get("OPTIONS", "preview_window")
    logger.info(f"Preview window mode from config: {preview_window}")

all_mouth_landmarks = []

# Global models (to be loaded later)
model = detector = detector_model = None


def do_load(checkpoint_path: str) -> None:
    """Load the Wav2Lip model and initialize the face detector.

    This function sets the global variables: model, detector, and detector_model.

    Args:
        checkpoint_path (str): Path to the Wav2Lip model checkpoint.
    """
    global model, detector, detector_model
    logger.info(f"Loading Wav2Lip model from checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path)
    detector = RetinaFace(gpu_id=gpu_id, model_path="checkpoints/mobilenet.pth", network="mobilenet")
    detector_model = detector.model
    logger.info("Model and face detector loaded successfully.")


def face_rect(images: list) -> "Generator[tuple, None, None]":
    """Detect faces in a batch of images and yield bounding boxes.

    Splits images into batches and uses the detector to obtain face bounding boxes.
    If a face is detected in an image, returns the bounding box of the first detected face;
    otherwise, yields the previous bounding box.

    Args:
        images (list): List of images (numpy arrays).

    Yields:
        tuple or None: A tuple (x1, y1, x2, y2) representing the bounding box, or None if not detected.
    """
    face_batch_size: int = 8
    num_batches: int = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size: (i + 1) * face_batch_size]
        all_faces = detector(batch)  # Returns list of faces for each image in the batch.
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret


def create_tracked_mask(img: np.ndarray, original_img: np.ndarray) -> tuple:
    """Create a tracked mask based on mouth detection using face landmarks.

    Uses the global predictor and mouth_detector to detect the mouth region and create a mask.
    If no face is detected, uses the last successful mask.

    Args:
        img (np.ndarray): Processed image (BGR format).
        original_img (np.ndarray): Original image (BGR format).

    Returns:
        tuple: (output image as np.ndarray, mask as PIL Image or None)
    """
    global kernel, last_mask, x, y, w, h

    # Convert images from BGR to RGB in-place.
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    # Detect face using mouth_detector.
    faces = mouth_detector(img)
    if len(faces) == 0:
        if last_mask is not None:
            last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
            mask = last_mask  # Use last successful mask.
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
    else:
        face = faces[0]
        shape = predictor(img, face)
        # Extract mouth landmarks (points 48 to 67).
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        # Calculate bounding box for the mouth.
        x, y, w, h = cv2.boundingRect(mouth_points)
        # Kernel size proportional to bounding box.
        kernel_size = int(max(w, h) * args.mask_dilation)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Create binary mask for the mouth region.
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        last_mask = mask

    # Dilate the mask.
    dilated_mask = cv2.dilate(mask, kernel)
    # Compute distance transform.
    dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
    # Threshold to create binary mask.
    _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
    masked_diff = masked_diff.astype(np.uint8)
    # Ensure blur size is odd.
    blur = args.mask_feathering
    if blur % 2 == 0:
        blur += 1
    blur = int(max(w, h) * blur)
    if blur % 2 == 0:
        blur += 1
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

    # Convert arrays to PIL Images.
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)
    mask_img = Image.fromarray(masked_diff)
    # Ensure all images are the same size.
    assert input1.size == input2.size == mask_img.size
    # Composite images using the mask.
    input2.paste(input1, (0, 0), mask_img)
    output_img = np.array(input2)
    cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB, output_img)
    return output_img, mask_img


def create_mask(img: np.ndarray, original_img: np.ndarray) -> tuple:
    """Create a mask for the mouth region based on face detection.

    If a previous mask exists, it is reused; otherwise, detects the face and computes the mask.

    Args:
        img (np.ndarray): Processed image (BGR format).
        original_img (np.ndarray): Original image (BGR format).

    Returns:
        tuple: (output image as np.ndarray, mask as PIL Image or None)
    """
    global kernel, last_mask, x, y, w, h

    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    if last_mask is not None:
        last_mask_np = np.array(last_mask)
        last_mask_np = cv2.resize(last_mask_np, (img.shape[1], img.shape[0]))
        mask = last_mask_np
        mask = Image.fromarray(mask)
    else:
        faces = mouth_detector(img)
        if len(faces) == 0:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)
            mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            x, y, w, h = cv2.boundingRect(mouth_points)
            kernel_size = int(max(w, h) * args.mask_dilation)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)
            dilated_mask = cv2.dilate(mask, kernel)
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)
            if args.mask_feathering != 0:
                blur = args.mask_feathering
                blur = int(max(w, h) * blur)
                if blur % 2 == 0:
                    blur += 1
                masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
            mask = Image.fromarray(masked_diff)
            last_mask = mask

    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)
    mask = mask.resize(input1.size)
    assert input1.size == input2.size == mask.size
    input2.paste(input1, (0, 0), mask)
    output_img = np.array(input2)
    cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB, output_img)
    return output_img, mask


def get_smoothened_boxes(boxes: np.ndarray, T: int) -> np.ndarray:
    """Smooth the bounding boxes over a temporal window.

    Args:
        boxes (np.ndarray): Array of bounding boxes.
        T (int): Temporal window size for smoothing.

    Returns:
        np.ndarray: Smoothed bounding boxes.
    """
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images: list, results_file: str = "last_detected_face.pkl") -> list:
    """Detect faces in a sequence of images and cache results.

    If a cache file exists, loads detection results from it.
    Otherwise, runs face detection on all images and caches the results.

    Args:
        images (list): List of images (numpy arrays).
        results_file (str, optional): Path to cache file. Defaults to "last_detected_face.pkl".

    Returns:
        list: List of detected face regions.
    """
    if os.path.exists(results_file):
        logger.info("Using cached face detection data.")
        with open(results_file, "rb") as f:
            return pickle.load(f)

    results = []
    pady1, pady2, padx1, padx2 = args.pads

    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, rect in tqdm_partial(
            zip(images, face_rect(images)),
            total=len(images),
            desc="Detecting face in every frame",
            ncols=100,
    ):
        if rect is None:
            cv2.imwrite("temp/faulty_frame.jpg", image)
            raise ValueError("Face not detected! Ensure the video contains a face in all frames.")
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    return results


def datagen(frames: list, mels: list):
    """Generator that yields batches of images and mel spectrograms for inference.

    Uses global variable args for configuration.

    Args:
        frames (list): List of video frames (numpy arrays).
        mels (list): List of mel spectrogram chunks (numpy arrays).

    Yields:
        tuple: (img_batch, mel_batch, frame_batch, coords_batch)
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    logger.info("Generating data batches for inference.")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        logger.info("Using specified bounding box instead of face detection.")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch = np.asarray(img_batch)
        mel_batch = np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch


# Global constant for mel step size.
mel_step_size: int = 16


def _load(checkpoint_path: str):
    """Load a PyTorch checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        Any: Loaded checkpoint.
    """
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def main() -> None:
    """Main function to run inference for lip-sync video generation.

    This function processes input video/audio, computes mel spectrograms,
    runs inference with the Wav2Lip model, applies optional upscaling and mask blending,
    and writes the output video.
    """
    args.img_size = 96
    frame_number: int = 11

    # Determine if the input face is a static image.
    if os.path.isfile(args.face) and args.face.split(".")[1].lower() in ["jpg", "png", "jpeg"]:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")
    elif args.face.split(".")[1].lower() in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        if args.fullres != 1:
            logger.info("Resizing video...")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    # Convert audio to .wav if needed.
    if not args.audio.endswith(".wav"):
        logger.info("Converting audio to .wav format...")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-i", args.audio, "temp/temp.wav"
        ])
        args.audio = "temp/temp.wav"

    logger.info("Analyzing audio...")
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel spectrogram contains NaN! Consider adding a small epsilon noise to the wav file and try again.")

    mel_chunks = []
    mel_idx_multiplier: float = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    full_frames = full_frames[: len(mel_chunks)]
    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
    logger.info(f"{len(full_frames)} frames to process")
    batch_size = args.wav2lip_batch_size
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks)
    else:
        gen = datagen(full_frames.copy(), mel_chunks)

    out = None
    run_params = None
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen,
                 total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
                 desc="Processing Wav2Lip",
                 ncols=100)):
        if i == 0:
            if not args.quality == "Fast":
                logger.info(f"Mask dilation: {args.mask_dilation}, feathering: {args.mask_feathering}")
                if not args.quality == "Improved":
                    logger.info(f"Loading super-resolution model: {args.sr_model}")
                    run_params = load_sr()
            logger.info("Starting inference...")
            frame_h, frame_w = full_frames[0].shape[:-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            if str(args.debug_mask) == "True":
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]
            if args.quality == "Enhanced":
                p = upscale(p, run_params)
            if args.quality in ["Enhanced", "Improved"]:
                if str(args.mouth_tracking) == "True":
                    p, last_mask = create_tracked_mask(p, cf)
                else:
                    p, last_mask = create_mask(p, cf)
            f[y1:y2, x1:x2] = p

            if not g_colab_flag:
                if preview_window == "Face":
                    cv2.imshow("face preview - press Q to abort", p)
                elif preview_window == "Full":
                    cv2.imshow("full preview - press Q to abort", f)
                elif preview_window == "Both":
                    cv2.imshow("face preview - press Q to abort", p)
                    cv2.imshow("full preview - press Q to abort", f)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    exit()
            if str(args.preview_settings) == "True":
                cv2.imwrite("temp/preview.jpg", f)
                if not g_colab_flag:
                    cv2.imshow("preview - press Q to close", f)
                    if cv2.waitKey(-1) & 0xFF == ord("q"):
                        exit()
            else:
                out.write(f)

    cv2.destroyAllWindows()
    if out is not None:
        out.release()

    if str(args.preview_settings) == "False":
        logger.info("Converting to final video...")
        subprocess.check_call([
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", "temp/result.mp4",
            "-i", args.audio,
            "-c:v", "libx264",
            args.outfile
        ])
        logger.info(f"Final video saved to {args.outfile}")


if __name__ == "__main__":
    args = parser.parse_args()
    do_load(args.checkpoint_path)
    main()
