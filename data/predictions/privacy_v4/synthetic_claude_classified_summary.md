# ModernBERT test summary: `synthetic_claude_classified.csv`

- Model run: `privacy_v4` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `privacy_leak_level`
- max_length: 2048  |  GPU: NVIDIA A10  |  3.8s (65.9 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 109 | 43.6% |
| MEDIUM | 83 | 33.2% |
| HIGH | 58 | 23.2% |

Mean confidence: 0.977

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.7280**
- **Macro F1: 0.6711**  |  Weighted F1: 0.7707
- Macro precision: 0.7184  |  Macro recall: 0.7386
- Mean confidence when correct: 0.990  |  when wrong: 0.943

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.982 | 0.775 | 0.866 | 138 | 109 |
| MEDIUM | 0.277 | 0.821 | 0.414 | 28 | 83 |
| HIGH | 0.897 | 0.619 | 0.732 | 84 | 58 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 107 | 28 | 3 |
| **MEDIUM** | 2 | 23 | 3 |
| **HIGH** | 0 | 32 | 52 |
