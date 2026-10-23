"""Grade transcripts for Czech sentence plausibility via the E-infra chat API.

- Empty text -> E, text containing 'Titulky' -> T (no model call).
- Otherwise: chat API with A/B/C/D prompt, first A/B/C/D letter in reply wins.
- B grades get a follow-up 'Attempt to fix this sentence' call -> 'fixed' field.
Writes one JSONL: {file, text, grade, raw_reply} — resumable.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from aicall import api_call

PROMPT_TEMPLATE = """Look at the following sentence in Czech. Reply only with one letter: A for 'this looks like a fully correct czech sentence', B for 'this doesn't look correct, but it's clear what it's supposed to say', C for 'this is likely incorrect', D for a special case, where the sentece is the word "pataka" repating. Repeating patterns are likely D.

Sentence: {text}"""

FIX_TEMPLATE = "Attempt to fix this sentence. Reply only with the corrected sentence and nothing else.\n\nSentence: {text}"


def pi_grade(text: str, model: str, retries: int) -> tuple[str, str]:
    for _ in range(retries):
        reply = api_call(PROMPT_TEMPLATE.format(text=text), model)
        if m := re.search(r"\b([ABCD])\b", reply):
            return m.group(1), reply[:200]
        time.sleep(2)
    return "C", f"GRADING_FAILED: {reply[:200]}"


def grade_and_fix(text: str, model: str, retries: int) -> tuple[str, str, str | None]:
    grade, reply = pi_grade(text, model, retries)
    if grade != "B":
        return grade, reply, None
    fixed = api_call(FIX_TEMPLATE.format(text=text), model)
    return grade, reply, fixed[:500]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", type=Path, required=True, help="all.jsonl from transcribe.py")
    p.add_argument("--output", type=Path, required=True, help="grades JSONL")
    p.add_argument("--model", default="deepseek")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.transcripts.read_text().splitlines()]
    done = {}
    if args.output.exists():
        for l in args.output.read_text().splitlines():
            d = json.loads(l)
            done[d["file"]] = d

    todo = [r for r in rows if r["file"] not in done]
    print(f"{len(rows)} transcripts, {len(todo)} to grade ({len(done)} cached)")

    out = args.output.open("a")
    graded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for r in todo:
            text = r["text"].strip()
            if not text or "Titulky" in text:
                g = "T" if "Titulky" in text else "E"
                d = {"file": r["file"], "text": r["text"], "grade": g, "raw_reply": "shortcut"}
                done[d["file"]] = d
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                continue
            futs[ex.submit(grade_and_fix, text, args.model, args.retries)] = r
        for fut in as_completed(futs):
            r = futs[fut]
            grade, reply, fixed = fut.result()
            d = {"file": r["file"], "text": r["text"], "grade": grade, "raw_reply": reply}
            if fixed is not None:
                d["fixed"] = fixed
            done[d["file"]] = d
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            out.flush()
            graded += 1
            if graded % 25 == 0:
                print(f"{graded}/{len(futs)} pi-graded")
    out.close()

    c = Counter(d["grade"] for d in done.values())
    print(f"done: {dict(c)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
