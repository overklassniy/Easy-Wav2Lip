import json
import os
import pickle
import re
import subprocess
from base64 import b64encode
from fractions import Fraction
from typing import Any, Tuple, Optional
from urllib.parse import urlparse

import dlib
import torch
from IPython.display import HTML, display
from torch.hub import download_url_to_file, get_dir

from models import Wav2Lip
from utils.logger import get_logger

# Get the project logger instance
logger = get_logger()

# Determine device: 'cuda', 'mps', or 'cpu'
device: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")


def get_video_details(filename: str) -> Tuple[int, int, float, float]:
    """
    Retrieves video details using ffprobe.

    This function runs ffprobe to extract video metadata and returns width, height, fps, and duration.

    Args:
        filename (str): Path to the video file.

    Returns:
        Tuple[int, int, float, float]: Width, height, fps, and duration of the video.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        filename,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = json.loads(result.stdout)

    # Find the video stream
    video_stream = next(
        stream for stream in info["streams"] if stream["codec_type"] == "video"
    )

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps = float(Fraction(video_stream["avg_frame_rate"]))
    duration = float(info["format"]["duration"])

    logger.debug(f"Video details for {filename}: width={width}, height={height}, fps={fps}, duration={duration}")
    return width, height, fps, duration


def show_video(file_path: str) -> None:
    """
    Displays a video in a Jupyter Notebook or Google Colab environment.

    This function reads the video file, encodes it in base64, and displays it using HTML.

    Args:
        file_path (str): Path to the video file.
    """
    try:
        with open(file_path, "rb") as f:
            mp4 = f.read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        width, _, _, _ = get_video_details(file_path)
        html_content = f"""
            <video controls width={min(width, 1280)}>
                <source src="{data_url}" type="video/mp4">
            </video>
            """
        display(HTML(html_content))
        logger.info(f"Displayed video from {file_path}")
    except Exception as e:
        logger.error(f"Error displaying video from {file_path}: {e}")


def format_time(seconds: float) -> str:
    """
    Formats seconds into a human-readable time string.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        formatted_time = f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        formatted_time = f"{minutes}m {secs}s"
    else:
        formatted_time = f"{secs}s"

    logger.debug(f"Formatted {seconds} seconds as {formatted_time}")
    return formatted_time


def _load(checkpoint_path: str) -> Any:
    """
    Loads a model checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        Any: Loaded checkpoint.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path: str) -> torch.nn.Module:
    """
    Loads the Wav2Lip model from a checkpoint file.

    If a cached file with a .pk1 extension exists, the model is loaded from it.
    Otherwise, the model is loaded from the checkpoint, state_dict keys are adjusted,
    the model is moved to the designated device, cached, and returned in eval mode.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: The loaded and initialized model.
    """
    folder, filename_with_extension = os.path.split(path)
    filename, _ = os.path.splitext(filename_with_extension)
    results_file = os.path.join(folder, filename + ".pk1")
    if os.path.exists(results_file):
        logger.info(f"Loading cached model from {results_file}")
        with open(results_file, "rb") as f:
            model = pickle.load(f)
        return model

    model = Wav2Lip()
    logger.info(f"Loading model from {path}")
    checkpoint = _load(path)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    # Cache the model in eval mode
    with open(results_file, "wb") as f:
        pickle.dump(model.eval(), f)
    logger.info(f"Model loaded and cached to {results_file}")
    return model.eval()


def get_input_length(filename: str) -> float:
    """
    Retrieves the duration of a media file using ffprobe.

    Args:
        filename (str): Path to the media file.

    Returns:
        float: Duration of the file in seconds.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = float(result.stdout)
    logger.debug(f"Input length for {filename}: {duration} seconds")
    return duration


def is_url(string: str) -> bool:
    """
    Checks if a string is a valid URL.

    Args:
        string (str): String to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    url_regex = re.compile(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$")
    valid = bool(url_regex.match(string))
    logger.debug(f"Checked URL {string}: {valid}")
    return valid


def load_predictor() -> None:
    """
    Loads and caches the face landmark predictor and face detector.

    Loads the shape predictor from a .dat file and the frontal face detector,
    then serializes them using pickle and saves them in the checkpoints directory.

    Returns:
        None
    """
    checkpoint_path = os.path.join("checkpoints", "shape_predictor_68_face_landmarks_GTX.dat")
    logger.info(f"Loading face predictor from {checkpoint_path}")
    predictor = dlib.shape_predictor(checkpoint_path)
    mouth_detector = dlib.get_frontal_face_detector()

    # Serialize the predictor and detector
    with open(os.path.join("checkpoints", "predictor.pkl"), "wb") as f:
        pickle.dump(predictor, f)
    with open(os.path.join("checkpoints", "mouth_detector.pkl"), "wb") as f:
        pickle.dump(mouth_detector, f)
    logger.info("Face predictor and mouth detector loaded and cached.")

    # Optionally remove the .dat file if it's no longer needed
    # os.remove(checkpoint_path)


def load_file_from_url(url: str, model_dir: Optional[str] = None, progress: bool = True, file_name: Optional[str] = None) -> str:
    """
    Downloads a file from a URL if it does not already exist locally.

    Args:
        url (str): URL to download the file from.
        model_dir (Optional[str]): Directory to save the file. If None, uses the default pytorch hub_dir.
        progress (bool): Whether to show the download progress.
        file_name (Optional[str]): Name of the downloaded file. If None, uses the filename from the URL.

    Returns:
        str: Absolute path to the downloaded file.
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    local_filename = os.path.basename(parts.path)
    if file_name is not None:
        local_filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, local_filename))
    if not os.path.exists(cached_file):
        logger.info(f"Downloading file from {url} to {cached_file}")
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    else:
        logger.info(f"File already exists at {cached_file}")
    return cached_file


def g_colab() -> bool:
    """
    Checks if the code is running in a Google Colab environment.

    Returns:
        bool: True if running in Google Colab, False otherwise.
    """
    try:
        import google.colab  # type: ignore
        logger.debug("Running in Google Colab environment.")
        return True
    except ImportError:
        logger.debug("Not running in Google Colab environment.")
        return False
