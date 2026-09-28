# ModernBERT test summary: `financial_pii_dataset.csv`

- Model run: `privacy_v4` (answerdotai/ModernBERT-base)
- Rows in file: 50  |  scored: 50  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `ground_truth`
- max_length: 2048  |  GPU: NVIDIA A10  |  3.6s (14.0 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 0 | 0.0% |
| MEDIUM | 0 | 0.0% |
| HIGH | 50 | 100.0% |

Mean confidence: 1.000

## Evaluation vs ground truth

- Rows evaluated: 50
- **Accuracy: 1.0000**
- **Macro F1: 1.0000**  |  Weighted F1: 1.0000
- Macro precision: 1.0000  |  Macro recall: 1.0000
- Mean confidence when correct: 1.000  |  when wrong: n/a

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.000 | 0.000 | 0.000 | 0 | 0 |
| MEDIUM | 0.000 | 0.000 | 0.000 | 0 | 0 |
| HIGH | 1.000 | 1.000 | 1.000 | 50 | 50 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 0 | 0 | 0 |
| **MEDIUM** | 0 | 0 | 0 |
| **HIGH** | 0 | 0 | 50 |
