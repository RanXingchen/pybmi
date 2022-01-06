import numpy as np
import pysptk
import torch
import soundfile as sf
import os
import pyworld as pw

from pysptk.synthesis import MLSADF, Synthesizer


def audio_synthesizer(mel: np.ndarray, pitch: np.ndarray, save_path: str,
                      file_name: str, hopsize: int, alpha: float = 0.35,
                      order: int = 25, sr: int = 22050, vocoder: str = 'SPTK',
                      ap: np.ndarray = None, frame_period: int = 10):
    """
    Synthesis the audio by MELs and pitch.

    Parameters
    ----------
    mel : ndarray
        The Mel-cepstrum.
    pitch : ndarray

    alpha : float, optional
        All-pass constant. Default is 0.35.

    """
    mel = mel.astype(np.float64)
    pitch = pitch.astype(np.float64)
    # Force the values smaller than 0 equal to 0.
    pitch[pitch < 0] = 0
    if vocoder == 'SPTK':
        excitation = pysptk.excite(np.ascontiguousarray(pitch), hopsize)
        b = pysptk.mc2b(mel, alpha)
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
            pysptk.mc2sp, 1, mel, alpha, fftlen
        )
        audio = pw.synthesize(f0, approximate_sp, decode_ap, sr, frame_period)

    audio = torch.tensor(audio).float()
    # Check NaN in reconstructed audio and replace it to 0
    audio[torch.isnan(audio)] = 0
    # Normalization the audio sequence to [-1, 1].
    normed_audio = (audio - audio.min()) / (audio.max() - audio.min()) * 2 - 1
    # Check again
    normed_audio[torch.isnan(normed_audio)] = 0
    # save the wave file
    sf.write(os.path.join(save_path, file_name + '.wav'), normed_audio, samplerate=sr)
    return normed_audio
