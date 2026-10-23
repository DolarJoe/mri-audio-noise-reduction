#!/bin/bash
set -x
.venv/bin/python denoise.py --output-dir denoised/hp_notch --num-sessions 10 --method hp,notch
.venv/bin/python transcribe.py --input-dir denoised/hp_notch --output-dir transcripts/hp_notch --num-sessions 10
.venv/bin/python grade.py --transcripts transcripts/hp_notch/all.jsonl --output grades/hp_notch.jsonl
.venv/bin/python compare.py --baseline grades/raw.jsonl --variant grades/hp_notch.jsonl --output reports/hp_notch_vs_raw.md
