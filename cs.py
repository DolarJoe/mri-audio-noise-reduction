"""Correlation subtraction (Inouye et al., JASA 135(3):1019-1022, 2014).

Time-domain CS: cross-correlation-align a noise-only reference to the
recording (single global lag, as in the paper) and subtract it.

Reference construction (clip mode): the last --cs-ref-clip seconds of the
recording are treated as noise-only, trimmed to an integer number of noise
periods, and tiled to full length. Tiling at integer periods is REQUIRED:
a 1.0 s tile of 0.98 s-periodic noise slips the noise phase by ~20 ms per
tile (~80 ms by clip end), which reads as desynchronization late in the
clip (measured: local alignment lags -961, -1, +960, +1920, +2880 samples).
"""
import numpy as np
from scipy.signal import correlate, correlation_lags


def detect_noise_period(y: np.ndarray, sr: int, lo: float = 0.3, hi: float = 2.0) -> float:
    """Dominant envelope periodicity in seconds (MRI gradient noise repeats)."""
    w = int(0.02 * sr)
    n = len(y) // w * w
    if n < 10 * w:
        return 0.98
    env = np.sqrt(np.mean(y[:n].reshape(-1, w) ** 2, axis=1))
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    if ac[0] <= 0:
        return 0.98
    ac /= ac[0]
    i0, i1 = int(lo / 0.02), int(hi / 0.02)
    k = int(np.argmax(ac[i0:i1])) + i0
    return k * 0.02


def _shift_data(data: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return data.copy()
    out = np.zeros_like(data)
    if lag < 0:
        out[: len(data) + lag] = data[-lag:]
    else:
        out[lag:] = data[: len(data) - lag]
    return out


def build_clip_reference(y: np.ndarray, sr: int, clip_s: float,
                         period_s: float | None = None) -> np.ndarray:
    """Last `clip_s` seconds, trimmed to an integer number of noise periods,
    tiled to the full signal length (phase-continuous)."""
    p = int(round((period_s or detect_noise_period(y, sr)) * sr))
    p = max(p, 1)
    clip_n = min(int(clip_s * sr), len(y))
    # keep whole periods from the end of the clip
    k = max(1, clip_n // p)
    seg = y[len(y) - clip_n: len(y) - clip_n + k * p]
    reps = int(np.ceil(len(y) / len(seg)))
    return np.tile(seg, reps)[: len(y)]


def correlation_subtraction(signal: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Paper CS: argmax of cross-correlation, shift reference, subtract."""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if len(noise) < len(signal):
        noise = np.concatenate([noise, np.zeros(len(signal) - len(noise))])
    C = correlate(noise, signal, mode="full", method="fft")
    lags = correlation_lags(len(noise), len(signal), mode="full")
    lag = int(lags[int(np.argmax(C))])
    return signal - _shift_data(noise, -lag)


def cs_denoise(y: np.ndarray, sr: int, *, cs_ref_clip: float = 4.0,
               cs_period: float | None = None) -> np.ndarray:
    ref = build_clip_reference(y, sr, cs_ref_clip, cs_period)
    return correlation_subtraction(y, ref)
