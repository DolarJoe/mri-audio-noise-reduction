"""Denoise MRI noise from speech WAVs. Judge only by resulting transcripts.

Methods (stackable, order = --method string):
  hp   : butterworth high-pass  (--highpass HZ, --hp-order)
  lp   : butterworth low-pass   (--lowpass HZ)
  nr   : noisereduce spectral gating (--prop-decrease, --stationary,
         --n-fft, --freq-mask-smooth-hz, --time-mask-smooth-ms,
         --noise-clip S = use first S seconds as noise profile)
  notch: auto-detected tonal notches (--notch-min-hz, --notch-max-hz,
         --notch-count, --notch-prom, --notch-q)

Example: denoise.py --method nr,hp --prop-decrease 0.9 --output-dir denoised/nr09

Pipeline use: import denoise() and call it with numpy audio plus keywords
matching the CLI dests, e.g. denoise(y, sr, method="hp", highpass_hz=300).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import noisereduce as nr
import scipy.signal as sg
import soundfile as sf

import trials

# CLI args consumed by file selection, not by the DSP chain.
SELECT_KEYS = {"input_dir", "output_dir", "files", "sessions", "num_sessions"}


def hp_filter(y, sr, cutoff, order):
    sos = sg.butter(order, cutoff, "high", fs=sr, output="sos")
    return sg.sosfiltfilt(sos, y)


def lp_filter(y, sr, cutoff, order):
    sos = sg.butter(order, cutoff, "low", fs=sr, output="sos")
    return sg.sosfiltfilt(sos, y)


def notch(y, sr, min_hz, max_hz, count, prom, q):
    f, P = sg.welch(y, sr, nperseg=8192)
    band = (f >= min_hz) & (f <= max_hz)
    peaks, props = sg.find_peaks(P[band], prominence=prom * P[band].max())
    freqs = f[band][peaks]
    taken = []
    for i in np.argsort(props["prominences"])[::-1]:
        if len(taken) >= count:
            break
        f0 = freqs[i]
        if any(abs(f0 - t) < 100 for t in taken):
            continue
        k = 1
        while k * f0 <= max_hz:  # notch the peak and its harmonics
            b, bb = sg.iirnotch(k * f0, q, fs=sr)
            y = sg.filtfilt(b, bb, y)
            k += 1
        taken.append(f0)
    return y


def denoise(y, sr, *, method="nr,hp", highpass_hz=400, lowpass_hz=7500,
            order=4, prop_decrease=1.0, stationary=False, n_fft=2048,
            freq_mask_smooth_hz=500, time_mask_smooth_ms=150, noise_clip=None,
            notch_min_hz=1000, notch_max_hz=8000, notch_count=3,
            notch_prom=0.05, notch_q=10):
    for m in method.split(","):
        m = m.strip()
        if m == "hp":
            y = hp_filter(y, sr, highpass_hz, order)
        elif m == "lp":
            y = lp_filter(y, sr, lowpass_hz, order)  # lp shares --hp-order
        elif m == "notch":
            y = notch(y, sr, notch_min_hz, notch_max_hz, notch_count,
                      notch_prom, notch_q)
        elif m == "nr":
            y_noise = y[: int(noise_clip * sr)] if noise_clip else None
            y = nr.reduce_noise(
                y=y, sr=sr, y_noise=y_noise, stationary=stationary,
                prop_decrease=prop_decrease, n_fft=n_fft,
                freq_mask_smooth_hz=freq_mask_smooth_hz,
                time_mask_smooth_ms=time_mask_smooth_ms)
        else:
            sys.exit(f"unknown method: {m}")
    peak = np.max(np.abs(y))
    return y / peak * 0.95 if peak > 0.95 else y


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    trials.add_selection_args(p)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--files", nargs="*", default=None,
                   help="specific wav filenames (takes precedence)")
    p.add_argument("--method", default="nr,hp")
    p.add_argument("--highpass", type=float, default=400, dest="highpass_hz", metavar="HZ")
    p.add_argument("--lowpass", type=float, default=7500, dest="lowpass_hz", metavar="HZ")
    p.add_argument("--hp-order", type=int, default=4, dest="order")
    p.add_argument("--prop-decrease", type=float, default=1.0)
    p.add_argument("--stationary", action="store_true")
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--freq-mask-smooth-hz", type=int, default=500)
    p.add_argument("--time-mask-smooth-ms", type=int, default=150)
    p.add_argument("--noise-clip", type=float, default=None)
    p.add_argument("--notch-min-hz", type=float, default=1000)
    p.add_argument("--notch-max-hz", type=float, default=8000)
    p.add_argument("--notch-count", type=int, default=3)
    p.add_argument("--notch-prom", type=float, default=0.05)
    p.add_argument("--notch-q", type=float, default=10)
    args = p.parse_args()

    if args.files:
        wavs = [args.input_dir / f for f in args.files]
    else:
        sessions = trials.select_sessions(trials.find_sessions(args.input_dir),
                                          args.sessions, args.num_sessions)
        wavs = trials.session_wavs(args.input_dir, sessions)

    trials.write_run_meta(args.output_dir,
                          {k: str(v) for k, v in vars(args).items()})
    dsp = {k: v for k, v in vars(args).items() if k not in SELECT_KEYS}

    for i, w in enumerate(wavs):
        out = args.output_dir / w.name
        if out.exists():
            continue
        y, sr = sf.read(w)
        yd = denoise(np.asarray(y, dtype=np.float64), sr, **dsp)
        sf.write(out, yd, sr)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(wavs)}")
    print(f"done: {len(wavs)} files -> {args.output_dir}")


if __name__ == "__main__":
    main()
