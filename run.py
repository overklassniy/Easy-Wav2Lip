"""
run.py - Main execution script for Easy-Wav2Lip

This script processes input video and audio files to produce a lip-synced video output.
It retrieves configuration from config.ini, validates input files, constructs output file
paths, and runs the Wav2Lip inference process. Batch processing of sequentially numbered files
is supported.
"""

import argparse
import configparser
import contextlib
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from typing import NoReturn

from IPython.display import Audio, display
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from utils.basic import format_time, get_input_length, get_video_details, show_video, g_colab

# Initialize warnings filter.
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize logger.
from utils.logger import get_logger

logger = get_logger()

# Set up argument parser.
parser = argparse.ArgumentParser(description="Easy-Wav2Lip main run file")
parser.add_argument(
    "--video_file",
    type=str,
    help="Input video file path",
    required=False,
    default=False,
)
parser.add_argument(
    "--vocal_file",
    type=str,
    help="Input audio file path",
    required=False,
    default=False,
)
parser.add_argument(
    "--output_folder",
    type=str,
    help="Output video file path",
    required=False,
    default=False,
)
args = parser.parse_args()

# Retrieve variables from config.ini.
config = configparser.ConfigParser()
config.read("config.ini")

# Determine input video file.
if args.video_file:
    video_file: str = args.video_file
else:
    video_file = config["OPTIONS"]["video_file"]

# Determine input vocal file.
if args.vocal_file:
    vocal_file: str = args.vocal_file
else:
    vocal_file = config["OPTIONS"]["vocal_file"]

# Determine output folder path
if args.output_folder:
    output_folder: str = args.output_folder
else:
    output_folder = os.getcwd() + '\\' + config["OPTIONS"]["output_folder"]
os.makedirs(output_folder, exist_ok=True)

# Retrieve additional configuration options.
quality: str = config["OPTIONS"]["quality"]
output_height: str = config["OPTIONS"]["output_height"]
wav2lip_version: str = config["OPTIONS"]["wav2lip_version"]
use_previous_tracking_data: str = config["OPTIONS"]["use_previous_tracking_data"]
nosmooth: bool = config.getboolean("OPTIONS", "nosmooth")
U: int = config.getint("PADDING", "U")
D: int = config.getint("PADDING", "D")
L: int = config.getint("PADDING", "L")
R: int = config.getint("PADDING", "R")
size: float = config.getfloat("MASK", "size")
feathering: int = config.getint("MASK", "feathering")
mouth_tracking: bool = config.getboolean("MASK", "mouth_tracking")
debug_mask: bool = config.getboolean("MASK", "debug_mask")
batch_process: bool = config.getboolean("OTHER", "batch_process")
output_suffix: str = config["OTHER"]["output_suffix"]
include_settings_in_suffix: bool = config.getboolean("OTHER", "include_settings_in_suffix")

if g_colab():
    preview_input: bool = config.getboolean("OTHER", "preview_input")
else:
    preview_input = False
preview_settings: bool = config.getboolean("OTHER", "preview_settings")
frame_to_preview: int = config.getint("OTHER", "frame_to_preview")

working_directory: str = os.getcwd()

# Record the start time of processing.
start_time: float = time.time()

# Clean up video and audio file paths.
video_file = video_file.strip('"')
vocal_file = vocal_file.strip('"')

# Validate video_file.
if video_file == "":
    logger.error("video_file cannot be blank")
    sys.exit(1)
if os.path.isdir(video_file):
    logger.error(f"{video_file} is a directory, you need to point to a file")
    sys.exit(1)
if not os.path.exists(video_file):
    logger.error(f"Could not find file: {video_file}")
    sys.exit(1)

# Set checkpoint path based on Wav2Lip version.
if wav2lip_version == "Wav2Lip_GAN":
    checkpoint_path: str = os.path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth")
else:
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip.pth")

# Determine resolution scale and output height.
resolution_scale: int = 1
res_custom: bool = False
if output_height == "half resolution":
    resolution_scale = 2
elif output_height == "full resolution":
    resolution_scale = 1
else:
    res_custom = True
    resolution_scale = 3

in_width, in_height, in_fps, in_length = get_video_details(video_file)
out_height: int = round(in_height / resolution_scale)

if res_custom:
    out_height = int(output_height)
fps_for_static_image: int = 30

if output_suffix == "" and not include_settings_in_suffix:
    logger.error("Current suffix settings will overwrite your input video! Please add a suffix or tick include_settings_in_suffix")
    sys.exit(1)

frame_to_preview = max(frame_to_preview - 1, 0)

# Construct output suffix if include_settings_in_suffix is enabled.
if include_settings_in_suffix:
    if wav2lip_version == "Wav2Lip_GAN":
        output_suffix = f"{output_suffix}_GAN"
    output_suffix = f"{output_suffix}_{quality}"
    if output_height != "full resolution":
        output_suffix = f"{output_suffix}_{out_height}"
    if nosmooth:
        output_suffix = f"{output_suffix}_nosmooth1"
    else:
        output_suffix = f"{output_suffix}_nosmooth0"
    if U != 0 or D != 0 or L != 0 or R != 0:
        output_suffix = f"{output_suffix}_pads-"
        if U != 0:
            output_suffix = f"{output_suffix}U{U}"
        if D != 0:
            output_suffix = f"{output_suffix}D{D}"
        if L != 0:
            output_suffix = f"{output_suffix}L{L}"
        if R != 0:
            output_suffix = f"{output_suffix}R{R}"
    if quality.lower() != "fast":
        output_suffix = f"{output_suffix}_mask-S{size}F{feathering}"
        if mouth_tracking:
            output_suffix = f"{output_suffix}_mt"
        if debug_mask:
            output_suffix = f"{output_suffix}_debug"
if preview_settings:
    output_suffix = f"{output_suffix}_preview"

# Compute padding strings.
rescaleFactor: str = str(round(1 // resolution_scale))
pad_up: str = str(round(U * resolution_scale))
pad_down: str = str(round(D * resolution_scale))
pad_left: str = str(round(L * resolution_scale))
pad_right: str = str(round(R * resolution_scale))

################################################################################
# Reconstruct input paths.

# For video file.
folder, filename_with_extension = os.path.split(video_file)
filename, file_type = os.path.splitext(filename_with_extension)
filenumber_match = re.search(r"\d+$", filename)
if filenumber_match:
    filenumber: str = str(filenumber_match.group())
    filenamenonumber: str = re.sub(r"\d+$", "", filename)
else:
    filenumber = ""
    filenamenonumber = filename

# For vocal file.
if vocal_file == "":
    vocal_file = video_file
else:
    if not os.path.exists(vocal_file):
        logger.error(f"Could not find file: {vocal_file}")
        sys.exit(1)
    if os.path.isdir(vocal_file):
        logger.error(f"{vocal_file} is a directory, you need to point to a file")
        sys.exit(1)
audio_folder, audio_filename_with_extension = os.path.split(vocal_file)
audio_filename, audio_file_type = os.path.splitext(audio_filename_with_extension)
audio_filenumber_match = re.search(r"\d+$", audio_filename)
if audio_filenumber_match:
    audio_filenumber: str = str(audio_filenumber_match.group())
    audio_filenamenonumber: str = re.sub(r"\d+$", "", audio_filename)
else:
    audio_filenumber = ""
    audio_filenamenonumber = audio_filename
################################################################################

process_failed: bool = False
temp_output: str = os.path.join(working_directory, "temp", "output.mp4")
temp_folder: str = os.path.join(working_directory, "temp")

last_input_video: str = ""
last_input_audio: str = ""


def main() -> NoReturn:
    """Main function to execute the Easy-Wav2Lip processing pipeline."""
    global filenumber, audio_filenumber, last_input_video, last_input_audio, process_failed

    while True:
        # Construct input video and audio paths.
        input_video: str = os.path.join(folder, filenamenonumber + filenumber + file_type)
        input_videofile: str = os.path.basename(input_video)
        input_audio: str = os.path.join(audio_folder, audio_filenamenonumber + audio_filenumber + audio_file_type)
        input_audiofile: str = os.path.basename(input_audio)

        # Determine output filename.
        if filenamenonumber + filenumber != audio_filenamenonumber + audio_filenumber:
            output_filename: str = f"{filenamenonumber}{filenumber}_{audio_filenamenonumber}{audio_filenumber}"
        else:
            output_filename = filenamenonumber + filenumber

        output_video: str = os.path.join(output_folder, output_filename + output_suffix + ".mp4")
        output_video = os.path.normpath(output_video)

        # Prepare temporary folder.
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        os.makedirs("temp", exist_ok=True)

        # Preview inputs if enabled.
        if preview_input:
            logger.info("Previewing input files.")
            logger.info("Input video:")
            show_video(input_video)
            if vocal_file != "":
                logger.info("Input audio:")
                display(Audio(input_audio))
            else:
                logger.info(f"Using {input_videofile} for audio.")

        last_input_video = input_video
        last_input_audio = input_audio
        shutil.copy(input_video, temp_folder)
        shutil.copy(input_audio, temp_folder)

        # Rename temp video file to include padding info.
        temp_input_video: str = os.path.join(temp_folder, input_videofile)
        renamed_temp_input_video: str = os.path.join(temp_folder, f"{U}{D}{L}{R}{input_videofile}")
        shutil.copy(temp_input_video, renamed_temp_input_video)
        temp_input_video = renamed_temp_input_video
        temp_input_videofile: str = os.path.basename(renamed_temp_input_video)
        temp_input_audio: str = os.path.join(temp_folder, input_audiofile)

        # Trim video if it's longer than the audio.
        video_length: float = get_input_length(temp_input_video)
        audio_length: float = get_input_length(temp_input_audio)
        if preview_settings:
            # For preview, process only a short clip.
            global batch_process
            batch_process = False
            preview_length_seconds: int = 1
            converted_preview_frame: float = frame_to_preview / in_fps
            preview_start_time: float = min(converted_preview_frame, video_length - preview_length_seconds)
            preview_video_path: str = os.path.join(
                temp_folder,
                f"preview_{preview_start_time}_{U}{D}{L}{R}{input_videofile}",
            )
            preview_audio_path: str = os.path.join(temp_folder, "preview_" + input_audiofile)
            subprocess.call([
                "ffmpeg",
                "-loglevel", "error",
                "-i", temp_input_video,
                "-ss", str(preview_start_time),
                "-to", str(preview_start_time + preview_length_seconds),
                "-c", "copy",
                preview_video_path,
            ])
            subprocess.call([
                "ffmpeg",
                "-loglevel", "error",
                "-i", temp_input_audio,
                "-ss", str(preview_start_time),
                "-to", str(preview_start_time + preview_length_seconds),
                "-c", "copy",
                preview_audio_path,
            ])
            temp_input_video = preview_video_path
            temp_input_audio = preview_audio_path

        if video_length > audio_length:
            trimmed_video_path: str = os.path.join(temp_folder, "trimmed_" + temp_input_videofile)
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    ffmpeg_extract_subclip(temp_input_video, 0, audio_length, targetname=trimmed_video_path)
            temp_input_video = trimmed_video_path

        # Check if face detection has already been performed on this clip.
        last_detected_face: str = os.path.join(working_directory, "last_detected_face.pkl")
        if os.path.isfile("last_file.txt"):
            with open("last_file.txt", "r") as file:
                last_file: str = file.readline().strip()
            if last_file != temp_input_video or use_previous_tracking_data == "False":
                if os.path.isfile(last_detected_face):
                    os.remove(last_detected_face)

        logger.info(
            f"Processing{' preview of' if preview_settings else ''} {input_videofile} using {input_audiofile} for audio"
        )

        # Build the command to execute Wav2Lip inference.
        cmd = [
            sys.executable,
            "inference.py",
            "--face", temp_input_video,
            "--audio", temp_input_audio,
            "--outfile", temp_output,
            "--pads", pad_up, pad_down, pad_left, pad_right,
            "--checkpoint_path", checkpoint_path,
            "--out_height", str(out_height),
            "--fullres", str(resolution_scale),
            "--quality", quality,
            "--mask_dilation", str(size),
            "--mask_feathering", str(feathering),
            "--nosmooth", str(nosmooth),
            "--debug_mask", str(debug_mask),
            "--preview_settings", str(preview_settings),
            "--mouth_tracking", str(mouth_tracking),
        ]
        logger.info("Executing Wav2Lip inference command.")
        subprocess.run(cmd)

        if preview_settings:
            preview_image_path: str = os.path.join(temp_folder, "preview.jpg")
            if os.path.isfile(preview_image_path):
                logger.info(f"Preview successful! Check out {preview_image_path}")
                with open("last_file.txt", "w") as f:
                    f.write(temp_input_video)
                end_time: float = time.time()
                elapsed_time: float = end_time - start_time
                logger.info(f"Execution time: {format_time(elapsed_time)}")
                break
            else:
                logger.error("Processing failed during preview! Please check GitHub issues: https://github.com/overklassniy/Easy-Wav2Lip/issues")
                sys.exit(1)

        if os.path.isfile(temp_output):
            if os.path.isfile(output_video):
                os.remove(output_video)
            shutil.copy(temp_output, output_video)
            with open("last_file.txt", "w") as f:
                f.write(temp_input_video)
            logger.info(f"{output_filename} successfully lip-synced! Output saved at: {output_video}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Execution time: {format_time(elapsed_time)}")
        else:
            logger.error("Processing failed! Please check GitHub issues: https://github.com/overklassniy/Easy-Wav2Lip/issues")
            process_failed = True

        if not batch_process:
            if process_failed:
                sys.exit(1)
            else:
                break

        # ------------------- Batch processing: update file numbers -------------------
        if filenumber == "" and audio_filenumber == "":
            logger.info("Files not set for batch processing")
            break

        if filenumber != "":
            match = re.search(r"\d+", filenumber)
            filenumber = f"{filenumber[:match.start()]}{int(match.group()) + 1:0{len(match.group())}d}"
        if audio_filenumber != "":
            match = re.search(r"\d+", audio_filenumber)
            audio_filenumber = f"{audio_filenumber[:match.start()]}{int(match.group()) + 1:0{len(match.group())}d}"

        # Reconstruct input paths for next iteration.
        input_video = os.path.join(folder, filenamenonumber + filenumber + file_type)
        input_videofile = os.path.basename(input_video)
        input_audio = os.path.join(audio_folder, audio_filenamenonumber + audio_filenumber + audio_file_type)
        input_audiofile = os.path.basename(input_audio)

        # Determine which files exist.
        if os.path.exists(input_video) and os.path.exists(input_audio):
            continue
        if os.path.exists(input_video) and input_video != last_input_video:
            if audio_filenumber != "":
                match = re.search(r"\d+", audio_filenumber)
                audio_filenumber = f"{audio_filenumber[:match.start()]}{int(match.group()) - 1:0{len(match.group())}d}"
            continue
        if os.path.exists(input_audio) and input_audio != last_input_audio:
            if filenumber != "":
                match = re.search(r"\d+", filenumber)
                filenumber = f"{filenumber[:match.start()]}{int(match.group()) - 1:0{len(match.group())}d}"
            continue

        logger.info("Finished processing all sequentially numbered files.")
        if process_failed:
            logger.error("Processing failed on at least one video.")
            sys.exit(1)
        else:
            break


if __name__ == "__main__":
    main()
