# Ground-truth verification (GROUND_TRUTH.csv, 7 handpicked files)

Metric: character error rate (CER) against the hand-picked reference sentences,
after normalizing diacritics, case, and punctuation. Lower is better.
Only variants with transcripts for all 7 files are comparable (raw, hp_notch,
cs_sng_c30); the nr* transcript dirs cover only file 1 and are shown for
completeness but excluded from the ranking.

## Mean CER (7 files)

| variant | mean CER |
|---|---|
| raw | **0.272** |
| hp_notch | 0.296 |
| cs_sng_c30 | 0.390 |

## Per-file CER

| file | GT | raw | hp_notch | cs_sng_c30 |
|---|---|---|---|---|
| 8482_s1_t3 | Pan trha papir | 0.000 | 0.000 | 0.071 ("Pantrhá" — missing space) |
| 8526_s1_t28 | Pan zameta uh zahradu listi | 0.148 | 0.185 | **0.148** |
| 8542_s1_t61 | Tady nejaky muz trha papir na pul | **0.030** | 0.091 | 0.121 (inserted "to") |
| 8549_s1_t63 | Muz bali krabice, asi se stehuje | 0.161 | 0.161 | **0.065** |
| 8556_s1_t26 | Slecna horoleze, asi venku | **0.120** | 0.280 | 0.520 (garbled) |
| 8561_s1_t? | Kuchar strouha syr | 0.444 | 0.556 | **1.444 — full hallucination** ("U každého z nich je nějaká země.") |
| 8562_s1_t? | Tancici par tanci na bale | 1.000 (empty) | 0.800 | **0.360** |

## Reading

- Confirms the grade-based finding: cs_sng_c30 does not beat raw overall, and
  here also trails hp_notch slightly on mean CER (0.390 vs 0.296), with raw best.
- cs_sng is **high-variance**: it rescues file 7 (raw produced nothing) and is
  closest-to-truth on file 4, but fully hallucinates on file 6 where raw got
  the content. The catastrophic losses are again consistent with the binary
  spectral gate (SNG step) distorting speech — same B→C/D→T mechanism as the
  726-file comparison.
- Ground truth is diacritic-free; comparison normalizes both sides.
