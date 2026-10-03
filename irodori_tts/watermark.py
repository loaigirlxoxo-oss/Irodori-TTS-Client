from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

IRODORI_WATERMARK_PAYLOAD = (73, 82, 68, 84, 83)  # "IRDTS"

# SilentCipher's 44.1k model only works at 44.1 kHz, while the codec emits 48 kHz.
#   resample : 48k -> 44.1k -> watermark -> 48k. The voice itself is resampled twice.
#   native44 : 48k -> 44.1k -> watermark, and the output stays 44.1 kHz (one resample).
#   delta    : watermark a 44.1k copy, take only the difference, resample that
#              difference to 48k and add it to the untouched 48k voice.
# The watermark payload is identical in all three; only what happens to the voice differs.
WATERMARK_MODES = ("resample", "native44", "delta")
_MODEL_SR = 44100
# The payload repeats every message_len (21) STFT frames of 2048 samples at 44.1 kHz.
# Long clips are watermarked in pieces so the watermark model's VRAM stays flat
# (30 s at once peaked at about 2.5 GiB and ran a 6 GB card out of memory).
# Each piece must start on a multiple of this period; pieces that start anywhere
# else restart the pattern out of phase and the result no longer decodes (measured).
_PAYLOAD_PERIOD = 21 * 2048
_CHUNK_PERIODS = 10     # about 9.75 s per piece (VRAM peak about 1.0 GiB)
_CONTEXT_PERIODS = 1    # extra audio on both sides so the piece edges match the whole-clip result


@dataclass(frozen=True)
class WatermarkSpec:
    mode: str = "resample"
    # How far below the signal the watermark sits (SilentCipher message_sdr, dB).
    # None keeps the model's own default (47 dB). Measured on four voices: 55 is the
    # weakest that still decodes on all of them; 60 and above no longer decodes.
    strength_db: float | None = None
    # delta only: keep the watermark inside (low, high) Hz. low <= 0 means no
    # high-pass. Narrow bands stop decoding on some voices, so this is for experiments.
    band_hz: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in WATERMARK_MODES:
            raise ValueError(f"Unsupported watermark mode={self.mode!r}. Expected one of: {WATERMARK_MODES}.")
        if self.band_hz is not None:
            low, high = self.band_hz
            if not (0 <= low < high):
                raise ValueError(f"Invalid watermark band={self.band_hz!r}.")
            if self.mode != "delta":
                raise ValueError("A watermark band can only be used with mode='delta'.")


def _resample(audio: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    if src == dst:
        return audio
    import torchaudio

    return torchaudio.functional.resample(audio.view(1, -1), src, dst).view(-1)


def _band_limit(audio: torch.Tensor, *, sample_rate: int, band_hz: tuple[float, float]) -> torch.Tensor:
    # Zero-phase 8th-order Butterworth, the same filter the band measurements used.
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    low, high = band_hz
    high = min(float(high), sample_rate / 2 * 0.999)
    if low > 0:
        sos = butter(8, [float(low), high], btype="band", fs=sample_rate, output="sos")
    else:
        sos = butter(8, high, btype="low", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, audio.detach().cpu().double().numpy())
    return torch.from_numpy(np.ascontiguousarray(filtered)).to(audio.dtype)


def _as_single_channel_vector(audio: torch.Tensor) -> torch.Tensor | None:
    squeezed = audio.detach().float().squeeze()
    if squeezed.ndim == 0 or squeezed.numel() == 0:
        return None
    if squeezed.ndim == 1:
        return squeezed
    return squeezed.reshape(-1)


def _match_original_rank(audio: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 2:
        return audio.reshape(1, -1)
    return audio.reshape(-1)


class SilentCipherWatermarker:
    def __init__(self, *, device: str, model_type: str = "44.1k") -> None:
        self.model = self._load_backend(device=device, model_type=model_type)

    @staticmethod
    def _load_backend(*, device: str, model_type: str):
        try:
            import silentcipher
        except ImportError:
            logger.warning(
                "SilentCipher package is unavailable; generated audio will not be watermarked."
            )
            return None

        try:
            return silentcipher.get_model(model_type=model_type, device=device)
        except Exception as exc:
            logger.warning(
                "SilentCipher model could not be loaded (%s); generated audio will not be "
                "watermarked.",
                exc,
            )
            return None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _encode(self, vector: torch.Tensor, payload: list[int],
                strength_db: float | None) -> torch.Tensor:
        """Watermark a 44.1 kHz clip in period-aligned pieces."""
        step = _CHUNK_PERIODS * _PAYLOAD_PERIOD
        ctx = _CONTEXT_PERIODS * _PAYLOAD_PERIOD
        n = vector.numel()
        out = torch.empty(n, dtype=torch.float32)
        for start in range(0, n, step):
            end = min(n, start + step)
            lo, hi = max(0, start - ctx), min(n, end + ctx)
            encoded, _ = self.model.encode_wav(
                vector[lo:hi].to(self.model.device),
                _MODEL_SR,
                payload,
                message_sdr=strength_db,
                calc_sdr=False,
            )
            piece = torch.as_tensor(encoded, dtype=torch.float32, device="cpu").reshape(-1)
            out[start:end] = piece[start - lo:start - lo + (end - start)]
        return out

    def encode_one(
        self,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        spec: WatermarkSpec = WatermarkSpec(),
        payload: Iterable[int] = IRODORI_WATERMARK_PAYLOAD,
    ) -> tuple[torch.Tensor, int]:
        """Watermark one clip. Returns (audio, sample_rate); native44 changes the rate."""
        if self.model is None:
            return audio, int(sample_rate)

        vector = _as_single_channel_vector(audio)
        if vector is None:
            return audio, int(sample_rate)

        sr = int(sample_rate)
        voice = vector.cpu()
        copy44 = _resample(voice, sr, _MODEL_SR)
        marked44 = self._encode(copy44, list(payload), spec.strength_db)
        if spec.mode == "native44":
            out, out_sr = marked44, _MODEL_SR
        elif spec.mode == "resample":
            out = _resample(marked44, _MODEL_SR, sr)
            out = torch.nn.functional.pad(out, (0, max(0, voice.numel() - out.numel())))[: voice.numel()]
            out_sr = sr
        else:  # delta
            diff = _resample(marked44 - copy44, _MODEL_SR, sr)
            diff = torch.nn.functional.pad(diff, (0, max(0, voice.numel() - diff.numel())))[: voice.numel()]
            if spec.band_hz is not None:
                diff = _band_limit(diff, sample_rate=sr, band_hz=spec.band_hz)
            out, out_sr = voice + diff, sr
        return _match_original_rank(out, reference=audio), out_sr

    def encode_batch(self, audios: list[torch.Tensor], *, sample_rate: int,
                     spec: WatermarkSpec = WatermarkSpec()) -> tuple[list[torch.Tensor], int]:
        if self.model is None:
            return audios, int(sample_rate)
        out, out_sr = [], int(sample_rate)
        for audio in audios:
            encoded, out_sr = self.encode_one(audio, sample_rate=sample_rate, spec=spec)
            out.append(encoded)
        return out, out_sr
