"""Transcribe WAV trials via E-infra Whisper endpoint.

Baseline + denoised transcription runner. Keeps every run's transcripts
under --output-dir for comparison.

API key resolution: --api-key > E_INFRA_API_TOKEN env > ~/.pi/agent/auth.json (E-infra).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import trials

API_URL = "https://llm.ai.e-infra.cz/v1/audio/transcriptions"


def get_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    import os
    if k := os.environ.get("E_INFRA_API_TOKEN"):
        return k
    auth = json.loads((Path.home() / ".pi/agent/auth.json").read_text())
    return auth.get("E-infra", auth.get("einfra"))["key"]


def transcribe_one(wav: Path, out_json: Path, *, model: str, language: str,
                   api_key: str, retries: int) -> tuple[Path, str | None]:
    if out_json.exists():
        return wav, None
    for attempt in range(retries):
        try:
            with wav.open("rb") as f:
                r = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"model": model, "language": language,
                          "response_format": "verbose_json"},
                    files={"file": (wav.name, f, "audio/wav")},
                    timeout=180,
                )
            if r.status_code == 200:
                data = r.json()
                out_json.write_text(json.dumps({
                    "file": wav.name, "model": model, "language": language,
                    "text": data.get("text", "").strip(),
                    "segments": data.get("segments"),
                }, ensure_ascii=False, indent=1))
                return wav, None
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            err = str(e)
        time.sleep(5 * (attempt + 1))
    return wav, err


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    trials.add_selection_args(p)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="e.g. transcripts/raw or transcripts/denoised_<params>")
    p.add_argument("--model", default="whisper-large-v3")
    p.add_argument("--language", default="cs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    sessions = trials.select_sessions(trials.find_sessions(args.input_dir),
                                      args.sessions, args.num_sessions)
    if not sessions:
        sys.exit("no sessions matched")

    wavs = trials.session_wavs(args.input_dir, sessions)
    trials.write_run_meta(args.output_dir, {
        "input_dir": str(args.input_dir), "sessions": sessions,
        "model": args.model, "language": args.language,
        "files": len(wavs),
    })

    outs = {w: args.output_dir / f"{w.stem}.json" for w in wavs}
    todo = [w for w in wavs if not outs[w].exists()]
    print(f"{len(wavs)} files, {len(todo)} to transcribe "
          f"({len(wavs) - len(todo)} cached)")

    api_key = get_api_key(args.api_key)
    done, errors = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(transcribe_one, w, outs[w], model=args.model,
                          language=args.language, api_key=api_key,
                          retries=args.retries) for w in todo]
        for fut in as_completed(futs):
            wav, err = fut.result()
            done += 1
            if err:
                errors.append((wav.name, err))
                print(f"ERROR {wav.name}: {err}")
            if done % 50 == 0:
                print(f"{done}/{len(todo)}")

    # combined summary
    rows = []
    for w in wavs:
        if not outs[w].exists():
            continue
        d = json.loads(outs[w].read_text())
        rows.append({"file": w.name, "text": d["text"]})
    (args.output_dir / "all.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"done: {len(rows)} transcripts -> {args.output_dir / 'all.jsonl'}"
          f", {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
