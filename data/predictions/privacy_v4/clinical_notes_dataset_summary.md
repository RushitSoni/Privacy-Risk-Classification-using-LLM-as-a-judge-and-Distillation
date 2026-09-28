# ModernBERT test summary: `clinical_notes_dataset.csv`

- Model run: `privacy_v4` (answerdotai/ModernBERT-base)
- Rows in file: 1602  |  scored: 1602  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `ground_truth`
- max_length: 2048  |  GPU: NVIDIA A10G  |  16.6s (96.4 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 0 | 0.0% |
| MEDIUM | 128 | 8.0% |
| HIGH | 1474 | 92.0% |

Mean confidence: 0.960

## Evaluation vs ground truth

- Rows evaluated: 1602
- **Accuracy: 0.9201**
- **Macro F1: 0.9584**  |  Weighted F1: 0.9584
- Macro precision: 1.0000  |  Macro recall: 0.9201
- Mean confidence when correct: 0.972  |  when wrong: 0.813

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.000 | 0.000 | 0.000 | 0 | 0 |
| MEDIUM | 0.000 | 0.000 | 0.000 | 0 | 128 |
| HIGH | 1.000 | 0.920 | 0.958 | 1602 | 1474 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 0 | 0 | 0 |
| **MEDIUM** | 0 | 0 | 0 |
| **HIGH** | 0 | 128 | 1474 |
