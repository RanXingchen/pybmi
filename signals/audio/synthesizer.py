import numpy as np
import pysptk
import soundfile as sf
import os
import pyworld as pw

from pysptk.synthesis import MLSADF, Synthesizer


def audio_synthesizer(mcep: np.ndarray,
                      pitch: np.ndarray,
                      save_path: str,
                      file_name: str,
                      hopsize: int,
                      alpha: float = 0.35,
                      order: int = 24,
                      sr: int = 22050,
                      vocoder: str = 'SPTK',
                      ap: np.ndarray = None):
    """
    Synthesis the audio by MELs and pitch.

    Parameters
    ----------
    mcep : ndarray
        The Mel-cepstrum.
    pitch : ndarray
        Pitch used to reconstruct audio.
    save_path : str
        The path to save the audio file.
    file_name : str
        File name to save.
    hopsize : int
        Hop size of the MCEP.
    alpha : float, optional
        All-pass constant. Default is 0.35.
    order : int, optional
        order of the MCEP. Default: 24.
    sr : int, optional
        The desire sampling rate of the synthesis audio.
    vocder : str, optional
        The choice of the vocoder. Default: 'SPTK'.
    ap : ndarray, optional
        Aperiodicity used to reconstruct audio when vocoder is 'WORLD'.
    """
    mcep = mcep.astype(np.float64)
    pitch = pitch.astype(np.float64)
    # Force the values smaller than 0 equal to 0.
    pitch[pitch < 0] = 0

    if vocoder == 'SPTK':
        excitation = pysptk.excite(np.ascontiguousarray(pitch), hopsize)
        b = pysptk.mc2b(mcep, alpha)
        synthesizer = Synthesizer(MLSADF(order=order, alpha=alpha), hopsize)
        audio = synthesizer.synthesis(excitation, b)
    elif vocoder == 'WORLD':
        # Aperiodicity is necessary.
        assert ap is not None
        ap = ap.astype(np.float64)

        # Calcuating f0 from the pitch.
        f0 = np.zeros_like(pitch, dtype=np.float64)
        f0[pitch != 0] = sr / pitch[pitch != 0]

        # Get the FFT size and decode the aperiodicity.
        fftlen = pw.get_cheaptrick_fft_size(sr)
        decode_ap = pw.decode_aperiodicity(ap, sr, fftlen)
        # Spectrogram
        approximate_sp = np.apply_along_axis(
            pysptk.mc2sp, 1, mcep, alpha, fftlen
        )
        # Calculate the frame period.
        frame_period = 1000 / int(sr / hopsize)
        audio = pw.synthesize(f0, approximate_sp, decode_ap, sr, frame_period)

    # Check NaN in reconstructed audio and replace it to 0
    audio[np.isnan(audio)] = 0
    # Normalization the audio sequence to [-1, 1].
    audio = (audio - np.min(audio)) /\
        (np.max(audio) - np.min(audio)) * 2 - 1
    # Check again
    audio[np.isnan(audio)] = 0

    # save the wave file
    sf.write(os.path.join(save_path, file_name + '.wav'), audio,
             samplerate=sr)
    return audio
