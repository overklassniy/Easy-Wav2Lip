import os
from typing import List, Dict, Any

from utils.logger import get_logger

# Get the project logger instance.
logger = get_logger()


def get_image_list(data_root: str, split: str) -> List[str]:
    """Retrieve a list of image file paths based on a text file list.

    Reads a file named "filelists/{split}.txt", where each line represents a relative image path.
    If a line contains spaces, only the first token is used.
    The relative paths are joined with the data_root.

    Args:
        data_root (str): The root directory where images are stored.
        split (str): The dataset split (e.g., "train", "val", "test").

    Returns:
        List[str]: A list of full paths to image files.
    """
    logger.info(f"Getting image list for split '{split}' from data root '{data_root}'.")
    filelist: List[str] = []
    filepath: str = "filelists/{}.txt".format(split)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if " " in line:
                line = line.split()[0]
            filelist.append(os.path.join(data_root, line))
    logger.debug(f"Retrieved {len(filelist)} image paths.")
    return filelist


class HParams:
    """Class for managing hyperparameters using a dictionary.

    Attributes:
        data (Dict[str, Any]): Dictionary storing hyperparameter names and values.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HParams with given hyperparameters.

        Args:
            **kwargs: Arbitrary keyword arguments representing hyperparameters.
        """
        self.data: Dict[str, Any] = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Retrieve a hyperparameter by name.

        Args:
            key (str): The hyperparameter name.

        Raises:
            AttributeError: If the hyperparameter does not exist.

        Returns:
            Any: The value of the requested hyperparameter.
        """
        if key not in self.data:
            raise AttributeError(f"'HParams' object has no attribute {key}")
        return self.data[key]

    def set_hparam(self, key: str, value: Any) -> None:
        """Set or update a hyperparameter.

        Args:
            key (str): The hyperparameter name.
            value (Any): The new value for the hyperparameter.
        """
        self.data[key] = value
        logger.debug(f"Hyperparameter '{key}' set to {value}.")

    def values(self) -> Dict[str, Any]:
        """Return all hyperparameters as a dictionary.

        Returns:
            Dict[str, Any]: The dictionary of all hyperparameters.
        """
        return self.data


# Default hyperparameters.
hparams: HParams = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality.
    rescale=True,  # Whether to rescale audio prior to preprocessing.
    rescaling_max=0.9,  # Rescaling factor.

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,
    n_fft=800,  # FFT window size; extra window size is filled with 0 paddings to match this parameter.
    hop_size=200,  # Hop size for STFT, e.g., for 16000Hz, 200 samples (~12.5 ms).
    win_size=800,  # Window size for STFT, e.g., for 16000Hz, 800 samples (~50 ms).
    sample_rate=16000,  # Audio sample rate.
    frame_shift_ms=None,  # Frame shift in milliseconds; if None, hop_size is used. Recommended: 12.5
    signal_normalization=True,  # Whether to normalize spectrograms.
    allow_clipping_in_normalization=True,  # If True, spectrogram values are clipped to avoid outliers.
    # Whether to scale mel-spectrograms symmetrically around 0.
    # Also multiplies the output range by 2, faster and cleaner convergence)
    symmetric_mels=True,
    max_abs_value=4.0,  # Maximum absolute value for normalized data.
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # Whether to apply a pre-emphasis filter.
    preemphasis=0.97,  # Pre-emphasis filter coefficient.
    min_level_db=-100,  # Minimum decibel level for spectrogram.
    ref_level_db=20,  # Reference decibel level.
    fmin=55,  # Minimum frequency for mel filter bank.
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise.
    # To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525]
    fmax=7600,  # Maximum frequency for mel filter bank.
    img_size=96,  # Image size for processing.
    fps=25,  # Frames per second.
    batch_size=16,  # Batch size for training.
    initial_learning_rate=1e-4,  # Initial learning rate.
    nepochs=200000000000000000,  # Number of epochs (set very high to allow manual stop).
    num_workers=16,  # Number of worker threads for data loading.
    checkpoint_interval=3000,  # Interval between saving model checkpoints.
    eval_interval=3000,  # Evaluation interval.
    save_optimizer_state=True,  # Whether to save the optimizer state.
    syncnet_wt=0.0,  # Initial weight for syncnet loss (will be updated later).
    syncnet_batch_size=64,  # Batch size for syncnet.
    syncnet_lr=1e-4,  # Learning rate for syncnet.
    syncnet_eval_interval=10000,  # Evaluation interval for syncnet.
    syncnet_checkpoint_interval=10000,  # Checkpoint interval for syncnet.
    disc_wt=0.07,  # Weight for discriminator loss.
    disc_initial_learning_rate=1e-4,  # Initial learning rate for discriminator.
)


def hparams_debug_string() -> str:
    """Generate a formatted string of all hyperparameters for debugging purposes.

    Excludes the 'sentences' hyperparameter if present.

    Returns:
        str: A string representation of all hyperparameters.
    """
    values = hparams.values()
    hp_lines = [f"  {name}: {values[name]}" for name in sorted(values) if name != "sentences"]
    debug_str = "Hyperparameters:\n" + "\n".join(hp_lines)
    logger.debug("Generated hyperparameters debug string.")
    return debug_str
