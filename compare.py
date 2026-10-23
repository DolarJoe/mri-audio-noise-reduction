"""Compare two grade files (default: raw vs a denoised variant).

Report: grade distribution shift, transitions, pataka (D) survival,
sample of notable transcript changes. Writes markdown to --output.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ORDER = ["A", "B", "C", "D", "T", "E"]
SPECIAL = {"D", "T", "E"}


def lp(f):
    return {json.loads(l)["file"]: json.loads(l) for l in Path(f).read_text().splitlines()}


def is_pataka(d):
    t = d["text"].lower()
    return d["grade"] == "D" or "pataka" in t or "poteka" in t or "po takhle" in t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="grades/raw.jsonl")
    p.add_argument("--variant", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    base, var = lp(args.baseline), lp(args.variant)
    files = sorted(set(base) & set(var))
    cb = Counter(b["grade"] for b in base.values())
    cv = Counter(v["grade"] for v in var.values())

    transitions = Counter((base[f]["grade"], var[f]["grade"]) for f in files)
    pat_b = [f for f in files if is_pataka(base[f])]
    pat_survived = sum(1 for f in pat_b if is_pataka(var[f]))
    improved = [f for f in files
                if ORDER.index(var[f]["grade"]) < ORDER.index(base[f]["grade"])
                and base[f]["grade"] not in SPECIAL and var[f]["grade"] not in SPECIAL]
    worsened = [f for f in files
                if ORDER.index(var[f]["grade"]) > ORDER.index(base[f]["grade"])
                and base[f]["grade"] not in SPECIAL and var[f]["grade"] not in SPECIAL]

    def ex(f):
        b, v = base[f], var[f]
        fx = f" -> \"{v['fixed']}\"" if "fixed" in v else ""
        return f"- `{f}`: [{b['grade']}] \"{b['text'][:60]}\" → [{v['grade']}] \"{v['text'][:60]}\"{fx}"

    lines = [
        f"# {Path(args.variant).stem} vs {Path(args.baseline).stem}",
        "",
        f"files compared: {len(files)}",
        "",
        "## Grade distribution (raw → variant)",
        "",
        "| grade | raw | variant |",
        "|---|---|---|",
    ] + [f"| {g} | {cb[g]} | {cv[g]} |" for g in ORDER] + [
        "",
        f"score (A=3,B=2,D&C... see below): raw={cb['A']*3+cb['B']*2+cb['D']}, "
        f"variant={cv['A']*3+cv['B']*2+cv['D']}  (A=3,B=2,D=1,C=1)",
        "",
        "## Transitions (non-pataka)",
        "",
        f"improved (to better grade): {len(improved)}",
        f"worsened: {len(worsened)}",
        f"D→not-D lost: {sum(1 for f in pat_b if not is_pataka(var[f]))}",
        f"pataka survival: {pat_survived}/{len(pat_b)}",
        "",
        "## Examples: improved",
        "",
    ] + [ex(f) for f in improved[:10]] + [
        "",
        "## Examples: worsened",
        "",
    ] + [ex(f) for f in worsened[:10]]
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
