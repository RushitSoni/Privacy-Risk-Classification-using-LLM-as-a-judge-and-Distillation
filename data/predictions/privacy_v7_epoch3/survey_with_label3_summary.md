# ModernBERT test summary: `survey_with_label3.csv`

- Model run: `privacy_v7_epoch3` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `label3`
- max_length: 2048  |  GPU: NVIDIA A10  |  8.4s (29.9 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 151 | 60.4% |
| MEDIUM | 82 | 32.8% |
| HIGH | 17 | 6.8% |

Mean confidence: 0.954

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.7280**
- **Macro F1: 0.6302**  |  Weighted F1: 0.7184
- Macro precision: 0.7215  |  Macro recall: 0.6221
- Mean confidence when correct: 0.962  |  when wrong: 0.933

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.841 | 0.876 | 0.858 | 145 | 151 |
| MEDIUM | 0.500 | 0.672 | 0.573 | 61 | 82 |
| HIGH | 0.824 | 0.318 | 0.459 | 44 | 17 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 127 | 18 | 0 |
| **MEDIUM** | 17 | 41 | 3 |
| **HIGH** | 7 | 23 | 14 |
