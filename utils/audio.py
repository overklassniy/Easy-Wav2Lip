import sys
from typing import Tuple, Any, Optional

import librosa
import numpy as np
import soundfile as sf  # Use soundfile instead of deprecated librosa.output.write_wav
from scipy import signal
from scipy.io import wavfile

from utils.logger import get_logger

sys.path.append("..")
from hparams import hparams as hp

# Get the project logger instance
logger = get_logger()


def load_wav(path: str, sr: int) -> np.ndarray:
    """
    Load a WAV file using librosa.

    Args:
        path (str): Path to the WAV file.
        sr (int): Sampling rate to load the audio.

    Returns:
        np.ndarray: Audio time series.
    """
    logger.info(f"Loading WAV file from: {path} with sample rate: {sr}")
    # librosa.load returns a tuple (audio, sample_rate); we only need the audio.
    audio, _ = librosa.load(path, sr=sr)
    return audio


def save_wav(wav: np.ndarray, path: str, sr: int) -> None:
    """
    Save a WAV file using scipy.io.wavfile.

    The audio is normalized to the range of int16 before saving.

    Args:
        wav (np.ndarray): Audio time series.
        path (str): Output file path.
        sr (int): Sampling rate.
    """
    # Normalize audio to int16 range
    normalized_wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    logger.info(f"Saving WAV file to: {path} with sample rate: {sr}")
    wavfile.write(path, sr, normalized_wav.astype(np.int16))


def save_wavenet_wav(wav: np.ndarray, path: str, sr: int) -> None:
    """
    Save a WAV file for Wavenet using soundfile.

    This function replaces the deprecated librosa.output.write_wav.

    Args:
        wav (np.ndarray): Audio time series.
        path (str): Output file path.
        sr (int): Sampling rate.
    """
    logger.info(f"Saving Wavenet WAV file to: {path} with sample rate: {sr}")
    sf.write(path, wav, sr)


def preemphasis(wav: np.ndarray, k: float, preemphasize: bool = True) -> np.ndarray:
    """
    Apply pre-emphasis filter to the audio signal.

    Args:
        wav (np.ndarray): Audio time series.
        k (float): Pre-emphasis coefficient.
        preemphasize (bool, optional): Flag to apply pre-emphasis. Defaults to True.

    Returns:
        np.ndarray: Filtered audio signal.
    """
    if preemphasize:
        logger.debug("Applying pre-emphasis filter.")
        return signal.lfilter([1, -k], [1], wav)
    logger.debug("Pre-emphasis not applied.")
    return wav


def inv_preemphasis(wav: np.ndarray, k: float, inv_preemphasize: bool = True) -> np.ndarray:
    """
    Apply inverse pre-emphasis filter to the audio signal.

    Args:
        wav (np.ndarray): Audio time series.
        k (float): Pre-emphasis coefficient.
        inv_preemphasize (bool, optional): Flag to apply inverse pre-emphasis. Defaults to True.

    Returns:
        np.ndarray: Inverse filtered audio signal.
    """
    if inv_preemphasize:
        logger.debug("Applying inverse pre-emphasis filter.")
        return signal.lfilter([1], [1, -k], wav)
    logger.debug("Inverse pre-emphasis not applied.")
    return wav


def get_hop_size() -> int:
    """
    Get the hop size for STFT.

    If hp.hop_size is not defined, it is computed from frame_shift_ms and sample_rate.

    Returns:
        int: Hop size.
    """
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None, "frame_shift_ms must be defined in hyperparameters."
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    logger.debug(f"Computed hop size: {hop_size}")
    return hop_size


def linearspectrogram(wav: np.ndarray) -> np.ndarray:
    """
    Compute the linear spectrogram of an audio signal.

    The audio signal is pre-emphasized, transformed via STFT, converted to dB,
    and normalized if specified in hyperparameters.

    Args:
        wav (np.ndarray): Audio time series.

    Returns:
        np.ndarray: Linear spectrogram.
    """
    logger.info("Computing linear spectrogram.")
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        logger.debug("Normalizing linear spectrogram.")
        return _normalize(S)
    return S


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """
    Compute the mel spectrogram of an audio signal.

    The audio signal is pre-emphasized, transformed via STFT, converted to mel scale,
    converted to dB, and normalized if specified in hyperparameters.

    Args:
        wav (np.ndarray): Audio time series.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    logger.info("Computing mel spectrogram.")
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        logger.debug("Normalizing mel spectrogram.")
        return _normalize(S)
    return S


def _lws_processor() -> Any:
    """
    Initialize and return an LWS processor.

    LWS (Local Weighted Sum) processor is used for time-frequency processing
    in speech signals.

    Returns:
        Any: An instance of LWS processor.
    """
    import lws  # Imported locally to avoid dependency issues if LWS is not used.
    logger.debug("Initializing LWS processor.")
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y: np.ndarray) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal.

    Depending on the hyperparameters, uses either LWS or librosa.stft.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        np.ndarray: STFT of the audio signal.
    """
    if hp.use_lws:
        logger.debug("Computing STFT using LWS processor.")
        # Using LWS; transpose result to match expected shape.
        return _lws_processor().stft(y).T
    else:
        logger.debug("Computing STFT using librosa.stft.")
        return librosa.stft(
            y=y,
            n_fft=hp.n_fft,
            hop_length=get_hop_size(),
            win_length=hp.win_size
        )


def num_frames(length: int, fsize: int, fshift: int) -> int:
    """
    Compute the number of time frames in a spectrogram.

    Args:
        length (int): Length of the audio signal.
        fsize (int): Frame size.
        fshift (int): Frame shift.

    Returns:
        int: Number of frames.
    """
    pad = fsize - fshift
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    logger.debug(f"Number of frames computed: {M}")
    return M


def pad_lr(x: np.ndarray, fsize: int, fshift: int) -> Tuple[int, int]:
    """
    Compute left and right padding for an audio signal.

    Args:
        x (np.ndarray): Audio time series.
        fsize (int): Frame size.
        fshift (int): Frame shift.

    Returns:
        Tuple[int, int]: Left and right padding values.
    """
    M = num_frames(len(x), fsize, fshift)
    pad = fsize - fshift
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    logger.debug(f"Padding computed: left={pad}, right={pad + r}")
    return pad, pad + r


def librosa_pad_lr(x: np.ndarray, fsize: int, fshift: int) -> Tuple[int, int]:
    """
    Compute padding for an audio signal using librosa's method.

    Args:
        x (np.ndarray): Audio time series.
        fsize (int): Frame size.
        fshift (int): Frame shift.

    Returns:
        Tuple[int, int]: Left and right padding values.
    """
    left = 0
    right = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    logger.debug(f"Librosa padding computed: left={left}, right={right}")
    return left, right


# Global mel basis matrix (initialized once)
_mel_basis: Optional[np.ndarray] = None


def _linear_to_mel(spectrogram: np.ndarray) -> np.ndarray:
    """
    Convert a linear spectrogram to a mel spectrogram.

    Args:
        spectrogram (np.ndarray): Linear spectrogram.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    global _mel_basis
    if _mel_basis is None:
        logger.info("Building mel basis matrix.")
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis() -> np.ndarray:
    """
    Build the mel filter bank.

    Returns:
        np.ndarray: Mel filter bank matrix.
    """
    assert hp.fmax <= hp.sample_rate // 2, "hp.fmax must be <= sample_rate/2"
    mel_basis = librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )
    logger.debug("Mel basis matrix built.")
    return mel_basis


def _amp_to_db(x: np.ndarray) -> np.ndarray:
    """
    Convert an amplitude spectrogram to dB-scaled spectrogram.

    Args:
        x (np.ndarray): Amplitude spectrogram.

    Returns:
        np.ndarray: dB-scaled spectrogram.
    """
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x: np.ndarray) -> np.ndarray:
    """
    Convert a dB-scaled spectrogram back to amplitude.

    Args:
        x (np.ndarray): dB-scaled spectrogram.

    Returns:
        np.ndarray: Amplitude spectrogram.
    """
    return np.power(10.0, (x) * 0.05)


def _normalize(S: np.ndarray) -> np.ndarray:
    """
    Normalize a spectrogram.

    Args:
        S (np.ndarray): Spectrogram to normalize.

    Returns:
        np.ndarray: Normalized spectrogram.
    """
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            normalized = np.clip(
                (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db))
                - hp.max_abs_value,
                -hp.max_abs_value,
                hp.max_abs_value,
            )
        else:
            normalized = np.clip(
                hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)),
                0,
                hp.max_abs_value,
            )
        logger.debug("Spectrogram normalized with clipping allowed.")
        return normalized

    # When clipping is not allowed, ensure values are within expected range.
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0, "Spectrogram values out of expected range."
    if hp.symmetric_mels:
        normalized = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        normalized = hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))
    logger.debug("Spectrogram normalized without clipping.")
    return normalized


def _denormalize(D: np.ndarray) -> np.ndarray:
    """
    Denormalize a spectrogram.

    Args:
        D (np.ndarray): Normalized spectrogram.

    Returns:
        np.ndarray: Denormalized spectrogram.
    """
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            denormalized = (
                                   (np.clip(D, -hp.max_abs_value, hp.max_abs_value) + hp.max_abs_value)
                                   * -hp.min_level_db
                                   / (2 * hp.max_abs_value)
                           ) + hp.min_level_db
        else:
            denormalized = (
                                   np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value
                           ) + hp.min_level_db
        logger.debug("Spectrogram denormalized with clipping allowed.")
        return denormalized

    if hp.symmetric_mels:
        denormalized = ((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db
    else:
        denormalized = (D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db
    logger.debug("Spectrogram denormalized without clipping.")
    return denormalized
