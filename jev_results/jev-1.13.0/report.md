# JEV zero-shot privacy-risk evaluation

- Model requested: `jev-1.13.0`; reported by API: `jev-1.13.0`
- Samples: 250 (valid outputs: 250, invalid/failed: 0; invalid counted as errors)
- Gold class counts in this CSV: LOW 145, MEDIUM 61, HIGH 44
- Prompt hash: `99649db37e0968a5`; CSV hash: `db4a3df9f1cff3ed`

## Overall

| Metric | Value |
|---|---|
| accuracy | 0.732 |
| macro_f1 | 0.599 |
| weighted_f1 | 0.711 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LOW | 0.848 | 0.924 | 0.884 | 145 |
| MEDIUM | 0.481 | 0.639 | 0.549 | 61 |
| HIGH | 0.909 | 0.227 | 0.364 | 44 |

## Confusion matrix (rows = human gold, cols = JEV)

| gold \ pred | LOW | MEDIUM | HIGH | INVALID |
|---|---|---|---|---|
| LOW | 134 | 11 | 0 | 0 |
| MEDIUM | 21 | 39 | 1 | 0 |
| HIGH | 3 | 31 | 10 | 0 |

## Does JEV's confidence track correctness?

- Mean q when correct: 0.876; when incorrect: 0.701
- Error-detection AUROC (errors positive, score 1-q): 0.809 (0.5 = no signal)

| q bin | n | accuracy |
|---|---|---|
| <0.6 | 32 | 0.406 |
| [0.6,0.7) | 28 | 0.571 |
| [0.7,0.8) | 35 | 0.543 |
| [0.8,0.9) | 42 | 0.643 |
| [0.9,0.95) | 20 | 0.750 |
| [0.95,0.99) | 17 | 1.000 |
| [0.99,1) | 13 | 1.000 |
| =1 | 63 | 1.000 |

**Threshold table (descriptive, post hoc):**

| tau | accepted n | coverage | acc if accepted | acc if escalated |
|---|---|---|---|---|
| 0.5 | 243 | 0.972 | 0.741 | 0.429 |
| 0.6 | 218 | 0.872 | 0.780 | 0.406 |
| 0.7 | 190 | 0.760 | 0.811 | 0.483 |
| 0.8 | 155 | 0.620 | 0.871 | 0.505 |
| 0.9 | 113 | 0.452 | 0.956 | 0.547 |
| 0.95 | 93 | 0.372 | 1.000 | 0.573 |
| 0.99 | 76 | 0.304 | 1.000 | 0.615 |

Confident errors (q>=0.9): 5 of 113 confident predictions. Breakdown: {'HIGH->MEDIUM': 3, 'LOW->MEDIUM': 1, 'MEDIUM->LOW': 1}

## Cost / latency

- Input tokens: 231,282; est. cost at $0.042/Mtok: $0.0097 (output tokens are free per TypeSafe docs; verify current pricing)
- Median per-call latency: 0.341 s (concurrent client, includes retries)

## Interpretation checklist

- AUROC near 0.5 or confident errors concentrated in MEDIUM? Confidence gating will not help here.
- Do NOT tune the prompt or a threshold on these 250 and then report them; split off a dev subset first.