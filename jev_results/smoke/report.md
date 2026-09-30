# JEV zero-shot privacy-risk evaluation

- Model requested: `jev-1.13.0`; reported by API: `jev-1.13.0`
- Samples: 5 (valid outputs: 5, invalid/failed: 0; invalid counted as errors)
- Gold class counts in this CSV: LOW 4, MEDIUM 1, HIGH 0
- Prompt hash: `99649db37e0968a5`; CSV hash: `db4a3df9f1cff3ed`

> WARNING: only 5 samples evaluated (smoke test / --limit). Do not report these numbers.

## Overall

| Metric | Value |
|---|---|
| accuracy | 0.600 |
| macro_f1 | 0.250 |
| weighted_f1 | 0.600 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LOW | 0.750 | 0.750 | 0.750 | 4 |
| MEDIUM | 0.000 | 0.000 | 0.000 | 1 |
| HIGH | 0.000 | 0.000 | 0.000 | 0 |

## Confusion matrix (rows = human gold, cols = JEV)

| gold \ pred | LOW | MEDIUM | HIGH | INVALID |
|---|---|---|---|---|
| LOW | 3 | 1 | 0 | 0 |
| MEDIUM | 1 | 0 | 0 | 0 |
| HIGH | 0 | 0 | 0 | 0 |

## Does JEV's confidence track correctness?

- Mean q when correct: 1.000; when incorrect: 0.715
- Error-detection AUROC (errors positive, score 1-q): 1.000 (0.5 = no signal)

| q bin | n | accuracy |
|---|---|---|
| [0.6,0.7) | 1 | 0.000 |
| [0.7,0.8) | 1 | 0.000 |
| =1 | 3 | 1.000 |

**Threshold table (descriptive, post hoc):**

| tau | accepted n | coverage | acc if accepted | acc if escalated |
|---|---|---|---|---|
| 0.5 | 5 | 1.000 | 0.600 | n/a |
| 0.6 | 5 | 1.000 | 0.600 | n/a |
| 0.7 | 4 | 0.800 | 0.750 | 0.000 |
| 0.8 | 3 | 0.600 | 1.000 | 0.000 |
| 0.9 | 3 | 0.600 | 1.000 | 0.000 |
| 0.95 | 3 | 0.600 | 1.000 | 0.000 |
| 0.99 | 3 | 0.600 | 1.000 | 0.000 |

Confident errors (q>=0.9): 0 of 3 confident predictions. Breakdown: none

## Cost / latency

- Input tokens: 3,961; est. cost at $0.042/Mtok: $0.0002 (output tokens are free per TypeSafe docs; verify current pricing)
- Median per-call latency: 0.612 s (concurrent client, includes retries)

## Interpretation checklist

- AUROC near 0.5 or confident errors concentrated in MEDIUM? Confidence gating will not help here.
- Do NOT tune the prompt or a threshold on these 250 and then report them; split off a dev subset first.