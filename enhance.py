import warnings
from typing import Any

from gfpgan import GFPGANer

from utils.logger import get_logger

# Suppress warnings to reduce log noise.
warnings.filterwarnings("ignore")

# Get the project logger instance.
logger = get_logger()


def load_sr() -> GFPGANer:
    """
    Load and initialize the GFPGAN super resolution model.

    This function creates an instance of GFPGANer with specified parameters.

    Returns:
        GFPGANer: An instance of the GFPGAN super resolution model.
    """
    logger.info("Initializing GFPGANer model with specified parameters.")
    run_params = GFPGANer(
        model_path="checkpoints/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    logger.debug("GFPGANer model initialized successfully.")
    return run_params


def upscale(image: Any, properties: GFPGANer) -> Any:
    """
    Enhance the given image using the GFPGANer model.

    This function applies enhancement on the image via the GFPGANer instance.

    Args:
        image (Any): Input image to enhance.
        properties (GFPGANer): An instance of GFPGANer with an initialized model.

    Returns:
        Any: The enhanced output image.
    """
    logger.info("Starting image enhancement process.")
    # The enhance method returns a tuple; we are interested in the third element (output).
    _, _, output = properties.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True
    )
    logger.info("Image enhancement completed.")
    return output
