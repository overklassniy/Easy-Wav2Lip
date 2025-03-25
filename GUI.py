import configparser
import os
import subprocess
import sys
import webbrowser
from typing import Any
from urllib.request import getproxies

import gradio as gr

from utils.logger import get_logger

proxies = getproxies()
os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = proxies["http"]
os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = proxies["https"]
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"

# Initialize logger
logger = get_logger()

try:
    with open("installed", "r") as file:
        version: str = file.read().strip()
except FileNotFoundError:
    logger.error("Easy-Wav2Lip does not appear to have installed correctly. "
                 "Please try to install it again: https://github.com/anothermartz/Easy-Wav2Lip/issues")
    input("Press Enter to exit...")
    exit()

logger.info("Starting Easy-Wav2Lip GUI using Gradio.")


def read_config() -> configparser.ConfigParser:
    """
    Read the configuration from config.ini.

    Returns:
        configparser.ConfigParser: The configuration object.
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    logger.debug("Configuration loaded from config.ini")
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """
    Save the updated configuration back to config.ini.

    Args:
        config (configparser.ConfigParser): The configuration object.
    """
    with open("config.ini", "w") as config_file:
        config.write(config_file)
    logger.info("Configuration saved to config.ini")


def start_easy_wav2lip(
        video_file: Any,
        vocal_file: Any,
        quality: str,
        output_height: str,
        wav2lip_version: str,
        use_previous_tracking_data: bool,
        nosmooth: bool,
        preview_window: str,
        pad_u: float,
        pad_d: float,
        pad_l: float,
        pad_r: float,
        mask_size: float,
        mask_feathering: float,
        mouth_tracking: bool,
        debug_mask: bool,
        batch_process: bool,
        output_suffix: str,
        include_settings_in_suffix: bool,
        preview_settings: bool,
        frame_to_preview: str,
) -> str:
    """
    Start the Easy-Wav2Lip process by saving configuration and directly launching run.py.

    Args:
        video_file (Any): The video file input (as provided by Gradio).
        vocal_file (Any): The audio file input (as provided by Gradio).
        quality (str): Quality setting.
        output_height (str): Output height setting.
        wav2lip_version (str): Version of Wav2Lip to use.
        use_previous_tracking_data (bool): Whether to keep previous face tracking data.
        nosmooth (bool): Flag to disable smoothing.
        preview_window (str): Preview window mode.
        pad_u (float): Padding up.
        pad_d (float): Padding down.
        pad_l (float): Padding left.
        pad_r (float): Padding right.
        mask_size (float): Mask size.
        mask_feathering (float): Mask feathering.
        mouth_tracking (bool): Whether to track mouth for mask.
        debug_mask (bool): Whether to enable debug mode for mask.
        batch_process (bool): Flag for batch processing.
        output_suffix (str): Suffix for the output file.
        include_settings_in_suffix (bool): Flag to include settings in suffix.
        preview_settings (bool): Flag to process only one frame (preview mode).
        frame_to_preview (str): Frame number to preview.

    Returns:
        str: Status message indicating that the configuration was saved and the process started.
    """
    logger.info("Starting Easy-Wav2Lip process with provided configuration.")
    config = read_config()

    # Update OPTIONS section
    config["OPTIONS"]["video_file"] = video_file.get("name", "") if isinstance(video_file, dict) else ""
    config["OPTIONS"]["vocal_file"] = vocal_file.get("name", "") if isinstance(vocal_file, dict) else ""
    config["OPTIONS"]["quality"] = quality
    config["OPTIONS"]["output_height"] = output_height
    config["OPTIONS"]["wav2lip_version"] = wav2lip_version
    config["OPTIONS"]["use_previous_tracking_data"] = str(use_previous_tracking_data)
    config["OPTIONS"]["nosmooth"] = str(nosmooth)
    config["OPTIONS"]["preview_window"] = preview_window

    # Update PADDING section
    config["PADDING"]["u"] = str(pad_u)
    config["PADDING"]["d"] = str(pad_d)
    config["PADDING"]["l"] = str(pad_l)
    config["PADDING"]["r"] = str(pad_r)

    # Update MASK section
    config["MASK"]["size"] = str(mask_size)
    config["MASK"]["feathering"] = str(mask_feathering)
    config["MASK"]["mouth_tracking"] = str(mouth_tracking)
    config["MASK"]["debug_mask"] = str(debug_mask)

    # Update OTHER section
    config["OTHER"]["batch_process"] = str(batch_process)
    config["OTHER"]["output_suffix"] = output_suffix
    config["OTHER"]["include_settings_in_suffix"] = str(include_settings_in_suffix)
    config["OTHER"]["preview_settings"] = str(preview_settings)
    config["OTHER"]["frame_to_preview"] = frame_to_preview

    save_config(config)

    # Directly launch run.py in a separate process without using a run trigger file.
    subprocess.Popen([
        sys.executable, "run.py",
        "--video_file", video_file,
        "--vocal_file", vocal_file
    ])
    logger.info("run.py process started directly.")
    return "Configuration saved and process started!"


def open_github_link() -> None:
    """
    Open the Easy-Wav2Lip GitHub page in the default web browser.
    """
    webbrowser.open("https://github.com/anothermartz/Easy-Wav2Lip?tab=readme-ov-file#advanced-tweaking")
    logger.info("Opened GitHub page for advanced tweaking.")


def build_interface() -> None:
    """
    Build and launch the Gradio interface for Easy-Wav2Lip configuration.
    """
    config = read_config()

    with gr.Blocks() as demo:
        gr.Markdown(f"# Easy-Wav2Lip GUI\n\n**Version:** {version}")

        # Column for video and audio upload with fixed dimensions
        with gr.Column(elem_classes="narrow-column"):
            video_file_input = gr.Video(label="Video File", interactive=True, sources=['upload'], height=480)
            vocal_file_input = gr.Audio(label="Audio File", interactive=True, sources=['upload'], type='filepath')

        # Column for settings
        with gr.Column(elem_classes="narrow-column"):
            quality_input = gr.Dropdown(
                label="Select Quality",
                choices=["Fast", "Improved", "Enhanced"],
                value=config["OPTIONS"].get("quality", "Improved")
            )
            output_height_input = gr.Dropdown(
                label="Output Height",
                choices=["half resolution", "full resolution"],
                value=config["OPTIONS"].get("output_height", "full resolution")
            )
            wav2lip_version_input = gr.Dropdown(
                label="Wav2Lip Version",
                choices=["Wav2Lip", "Wav2Lip_GAN"],
                value=config["OPTIONS"].get("wav2lip_version", "Wav2Lip")
            )
            use_previous_tracking_data_input = gr.Checkbox(
                label="Keep previous face tracking data",
                value=config["OPTIONS"].getboolean("use_previous_tracking_data", True)
            )
            nosmooth_input = gr.Checkbox(
                label="Disable smoothing face detection",
                value=config["OPTIONS"].getboolean("nosmooth", True)
            )
            preview_window_input = gr.Dropdown(
                label="Preview Window",
                choices=["Face", "Full", "Both", "None"],
                value=config["OPTIONS"].get("preview_window", "Face")
            )
            pad_u_input = gr.Number(
                label="Padding Up",
                value=float(config["PADDING"].get("u", 0)),
                precision=0
            )
            pad_d_input = gr.Number(
                label="Padding Down",
                value=float(config["PADDING"].get("d", 10)),
                precision=0
            )
            pad_l_input = gr.Number(
                label="Padding Left",
                value=float(config["PADDING"].get("l", 0)),
                precision=0
            )
            pad_r_input = gr.Number(
                label="Padding Right",
                value=float(config["PADDING"].get("r", 0)),
                precision=0
            )
            mask_size_input = gr.Number(
                label="Mask Size",
                value=float(config["MASK"].get("size", 2.5))
            )
            mask_feathering_input = gr.Number(
                label="Mask Feathering",
                value=float(config["MASK"].get("feathering", 2.5))
            )
            mouth_tracking_input = gr.Checkbox(
                label="Track mouth for mask",
                value=config["MASK"].getboolean("mouth_tracking", True)
            )
            debug_mask_input = gr.Checkbox(
                label="Highlight mask for debugging",
                value=config["MASK"].getboolean("debug_mask", True)
            )
            batch_process_input = gr.Checkbox(
                label="Batch Process",
                value=config["OTHER"].getboolean("batch_process", True)
            )
            output_suffix_input = gr.Textbox(
                label="Output File Suffix",
                value=config["OTHER"].get("output_suffix", "_Easy-Wav2Lip"),
                max_lines=1
            )
            include_settings_in_suffix_input = gr.Checkbox(
                label="Include settings in suffix",
                value=config["OTHER"].getboolean("include_settings_in_suffix", True)
            )
            preview_settings_input = gr.Checkbox(
                label="Process one frame only",
                value=config["OTHER"].getboolean("preview_settings", True)
            )
            frame_to_preview_input = gr.Textbox(
                label="Frame to Preview",
                value=config["OTHER"].get("frame_to_preview", "100"),
                max_lines=1
            )

        # Buttons: Advanced Tweaking and Start
        with gr.Row():
            github_button = gr.Button("Advanced Tweaking (Open GitHub)")
        with gr.Row():
            start_button = gr.Button("Start Easy-Wav2Lip", variant="primary")
        output_message = gr.Textbox(label="Status", interactive=False)

        github_button.click(fn=open_github_link)
        start_button.click(
            fn=start_easy_wav2lip,
            inputs=[
                video_file_input,
                vocal_file_input,
                quality_input,
                output_height_input,
                wav2lip_version_input,
                use_previous_tracking_data_input,
                nosmooth_input,
                preview_window_input,
                pad_u_input,
                pad_d_input,
                pad_l_input,
                pad_r_input,
                mask_size_input,
                mask_feathering_input,
                mouth_tracking_input,
                debug_mask_input,
                batch_process_input,
                output_suffix_input,
                include_settings_in_suffix_input,
                preview_settings_input,
                frame_to_preview_input,
            ],
            outputs=output_message,
        )

        # Launch the interface with parameters to suppress unnecessary messages and errors.
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            debug=False,
            share=False,
            quiet=True
        )


if __name__ == "__main__":
    build_interface()
