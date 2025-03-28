import os
from typing import NoReturn

from enhance import load_sr
from utils.basic import load_file_from_url, load_model, load_predictor
from utils.logger import get_logger

# Initialize logger
logger = get_logger()

# Global version string
version: str = "v9.0"


def download_and_initialize_models() -> None:
    """Download and initialize the Wav2Lip and GFPGAN models.

    This function downloads the required model files for Wav2Lip (both GAN and standard versions)
    and GFPGAN, then initializes them using the provided utility functions.
    """
    working_directory: str = os.getcwd()
    logger.info("Downloading Wav2Lip essentials")

    # Download mobilenet.pth
    load_file_from_url(
        url="https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/mobilenet.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="mobilenet.pth",
    )

    # Download Wav2Lip_GAN.pth and load the model
    load_file_from_url(
        url="https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/Wav2Lip_GAN.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="Wav2Lip_GAN.pth",
    )
    model_path_gan: str = os.path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth")
    model_gan = load_model(model_path_gan)
    logger.info("Wav2Lip_GAN model loaded successfully")

    # Download Wav2Lip.pth and load the model
    load_file_from_url(
        url="https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/Wav2Lip.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="Wav2Lip.pth",
    )
    model_path: str = os.path.join(working_directory, "checkpoints", "Wav2Lip.pth")
    model = load_model(model_path)
    logger.info("Wav2Lip model loaded successfully")

    # Download GFPGAN essentials and initialize the GFPGAN model
    logger.info("Downloading GFPGAN essentials")
    load_file_from_url(
        url="https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/GFPGANv1.4.pth",
        model_dir="checkpoints",
        progress=True,
        file_name="GFPGANv1.4.pth",
    )
    load_sr()  # Initialize GFPGAN model
    logger.info("GFPGAN model loaded successfully")


def initialize_face_detectors() -> None:
    """Download and initialize face detectors.

    This function downloads the shape predictor file used for face landmark detection and
    initializes the predictor.
    """
    logger.info("Initializing face detectors")
    load_file_from_url(
        url="https://github.com/overklassniy/Easy-Wav2Lip/releases/download/Prerequisites/shape_predictor_68_face_landmarks_GTX.dat",
        model_dir="checkpoints",
        progress=True,
        file_name="shape_predictor_68_face_landmarks_GTX.dat",
    )
    load_predictor()
    logger.info("Face predictor loaded successfully")


def write_installation_file(version_str: str) -> None:
    """Write a file to signify that the installation is complete.

    Args:
        version_str (str): The version string to write into the file.
    """
    with open("installed", "w") as f:
        f.write(version_str)
    logger.info("Installation complete. Installed version: " + version_str)


def main() -> NoReturn:
    """Main installation procedure for Easy-Wav2Lip.

    Downloads and initializes all required models and predictors, writes an installation marker file,
    and logs completion messages.
    """
    download_and_initialize_models()
    initialize_face_detectors()
    write_installation_file(version)
    logger.info("Installation complete!")
    logger.info("If you just updated from v8 - make sure to download the updated Easy-Wav2Lip.cmd too!")
    # Exit the script
    exit()


if __name__ == "__main__":
    main()
