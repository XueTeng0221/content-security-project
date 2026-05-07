# src/utils/audio_utils.py
import torch
import torchaudio
import torchaudio.transforms as T

def extract_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
) -> torch.Tensor:
    """
    Extract log Mel spectrogram from audio waveform.

    Args:
        audio: Audio tensor of shape [batch, samples]
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of Mel filter banks

    Returns:
        Log Mel spectrogram of shape [batch, n_mels, time]
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_spec = mel_transform(audio)
    log_mel_spec = torch.log(mel_spec + 1e-9)

    return log_mel_spec
