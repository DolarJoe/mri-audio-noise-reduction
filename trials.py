"""Shared trial selection for the pipeline: sessions, WAV globs, run metadata.

Used by denoise.py and transcribe.py; import these helpers from external
pipelines instead of re-implementing filename conventions.
"""
import argparse
import json
import re
import time
from pathlib import Path

RAW_DIR = Path("../Records_for_VUT_analysis")
SESSION_PAT = re.compile(r"(LILA_sub_\d+_session_\d+)_trial_\d+\.wav$")


def find_sessions(input_dir: Path) -> list[str]:
    return sorted({m.group(1) for f in input_dir.iterdir()
                   if (m := SESSION_PAT.match(f.name))})


def select_sessions(sessions: list[str], only=None, first_n=None) -> list[str]:
    if only:
        wanted = set(only)
        return [s for s in sessions if s in wanted]
    if first_n:
        return sessions[:first_n]
    return sessions


def session_wavs(input_dir: Path, sessions: list[str]) -> list[Path]:
    return sorted(f for s in sessions
                  for f in input_dir.glob(f"{s}_trial_*.wav"))


def add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-dir", type=Path, default=RAW_DIR,
                   help="directory of trial WAVs (default: raw records)")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="session prefixes, e.g. LILA_sub_8482_session_1")
    p.add_argument("--num-sessions", type=int, default=None,
                   help="take first N sessions (sorted)")


def write_run_meta(output_dir: Path, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_run.json").write_text(
        json.dumps(meta | {"time": time.strftime("%F %T")}, indent=1))
