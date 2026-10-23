"""Adaptive pipeline: best candidate per file + combined fixer.

Inputs: grades/raw.jsonl and grades/<variant>.jsonl (same file set).
Per file:
- pick better candidate (A > B > D > C; tie -> raw)
- if best grade is B: fixer call with BOTH candidates -> final text
- D/C/A: no fixer (C unfixable, A fine, D = pataka, keep as-is)

Outputs: adaptive JSONL (all decisions logged) + markdown summary.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from aicall import api_call

SCORE = {"A": 4, "B": 3, "D": 2, "C": 1, "E": 0, "T": 0}

COMBINED_FIX = """Two automatic transcriptions of the same Czech speech recording were made from different audio versions.
Transcription 1: {t1}
Transcription 2: {t2}
Attempt to fix the sentence, using whichever transcription looks more plausible. Reply only with the corrected Czech sentence and nothing else."""

SINGLE_FIX = "Attempt to fix this sentence. Reply only with the corrected sentence and nothing else.\n\nSentence: {t1}"



def junk(t: str) -> bool:
    return not t.strip() or "Titulky" in t


def load(f):
    return {json.loads(l)["file"]: json.loads(l) for l in Path(f).read_text().splitlines()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("grades/raw.jsonl"))
    p.add_argument("--variant", type=Path, default=Path("grades/hp_notch.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--model", default="deepseek")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    raw, var = load(args.raw), load(args.variant)
    files = sorted(set(raw) & set(var))
    done = {}
    if args.output.exists():
        done = {json.loads(l)["file"]: json.loads(l) for l in args.output.read_text().splitlines()}

    decisions = {}
    fix_todo = {}
    for f in files:
        if f in done:
            continue
        r, v = raw[f], var[f]
        chosen, other = (r, v) if SCORE[r["grade"]] >= SCORE[v["grade"]] else (v, r)
        d = {"file": f,
             "raw_grade": r["grade"], "raw_text": r["text"],
             "var_grade": v["grade"], "var_text": v["text"],
             "best_grade": chosen["grade"], "final_text": chosen["text"],
             "fixed": False}
        decisions[f] = d
        if chosen["grade"] == "B":
            fix_todo[f] = (chosen["text"], other["text"])

    print(f"{len(files)} files, {len(done)} cached, "
          f"{len(decisions)} new decisions, {len(fix_todo)} fixer calls")

    def fix_one(f):
        t1, t2 = fix_todo[f]
        if junk(t2) or t1.strip() == t2.strip():
            return f, api_call(SINGLE_FIX.format(t1=t1), args.model)
        return f, api_call(COMBINED_FIX.format(t1=t1, t2=t2), args.model)

    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(d, out):
        d.setdefault("final_grade", d["best_grade"])
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
        out.flush()

    with args.output.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for f, d in decisions.items():
            if f in fix_todo:
                futs[ex.submit(fix_one, f)] = f
            else:
                emit(d, out)
        for fut in as_completed(futs):
            f, fixed = fut.result()
            d = decisions[f]
            d["final_text"] = fixed[:500]
            d["fixed"] = True
            d["final_grade"] = "B*"
            emit(d, out)
            n += 1
            print(f"{n}/{len(fix_todo)} fixed", flush=True)

    all_d = list(done.values()) + [decisions[f] for f in decisions]
    cr = Counter(d["raw_grade"] for d in all_d)
    cf = Counter(d["final_grade"] for d in all_d)
    pat_raw = sum(1 for d in all_d if d["raw_grade"] == "D")
    pat_fin = sum(1 for d in all_d if d["final_grade"] == "D")
    rep = [
        f"# adaptive: {args.raw.stem} + {args.variant.stem}",
        "",
        f"files: {len(all_d)}",
        "",
        "| grade | raw | final |",
        "|---|---|---|",
    ] + [f"| {g} | {cr[g]} | {cf[g]} |" for g in ["A", "B", "B*", "C", "D", "E", "T"]] + [
        "",
        f"pataka (D): raw={pat_raw}, final={pat_fin}",
        f"fixer applied: {sum(1 for d in all_d if d['fixed'])}",
        "",
        "## Sample finals",
        "",
    ]
    for d in all_d[:12]:
        rep.append(f"- `{d['file']}` [{d['raw_grade']}/{d['var_grade']}→{d['final_grade']}] {d['final_text'][:70]}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(rep) + "\n")
    print("\n".join(rep[:14]))
    print(f"-> {args.output}, {args.report}")


if __name__ == "__main__":
    main()
