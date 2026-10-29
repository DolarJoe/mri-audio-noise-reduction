"""Spectral noise gating via classic SoX noisered (Inouye et al. 2014, SNG step).

Uses SoX v14.4.x (the paper bundled 14.4.1; the noisered algorithm is a binary
spectral gate: bins with log power below profile + amount*8 are zeroed, with
0.5/0.5 time smoothing). NOTE: coeff=0 is not identity — bins below the
profile are gated regardless.

SoX resolution order: $SOX_BIN > ../tools/sox1442/bin/sox > `which sox`.
The sox_ng fork is NOT equivalent (rewritten noisered).

Noise profile source: last --sng-ref-clip seconds of the recording
(noise-only tail), matching the reference used by the cs method.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_SOX_CANDIDATES = [
    os.environ.get("SOX_BIN"),
    str(Path(__file__).resolve().parent.parent / "tools" / "sox1442" / "bin" / "sox"),
    None,  # `which sox`
]


def _sox() -> str:
    for c in _SOX_CANDIDATES:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    p = shutil.which("sox")
    if p:
        return p
    raise RuntimeError("sox binary not found; set SOX_BIN or build SoX v14.4.2")


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sox failed: {' '.join(cmd)}\n{r.stderr}")


def spectral_noise_gating(y: np.ndarray, sr: int, noise_sample: np.ndarray,
                          coeff: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    noise_sample = np.asarray(noise_sample, dtype=np.float64)
    norm = max(np.max(np.abs(y)), np.max(np.abs(noise_sample))) / 0.8

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        prof_src, speech_src = td / "np.wav", td / "speech.wav"
        profile, cleaned = td / "p.prof", td / "clean.wav"
        sf.write(prof_src, noise_sample / norm, sr, subtype="PCM_16")
        sf.write(speech_src, y / norm, sr, subtype="PCM_16")
        _run([_sox(), str(prof_src), "-n", "noiseprof", str(profile)])
        _run([_sox(), str(speech_src), str(cleaned),
              "noisered", str(profile), f"{coeff:g}"])
        out, _ = sf.read(cleaned)

    res = np.zeros_like(y)
    n = min(len(y), len(out))
    res[:n] = out[:n]  # sox trims WINDOWSIZE/2=1024 samples at the end
    return res * norm


def sng_denoise(y: np.ndarray, sr: int, *, sng_coeff: float = 0.3,
                sng_ref_clip: float = 4.0) -> np.ndarray:
    from cs import detect_noise_period, build_clip_reference
    p = detect_noise_period(y, sr)
    ref = build_clip_reference(y, sr, sng_ref_clip, p)
    return spectral_noise_gating(y, sr, ref[: sr], sng_coeff)
