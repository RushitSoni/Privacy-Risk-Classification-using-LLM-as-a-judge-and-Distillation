# ModernBERT test summary: `survey_with_label3.csv`

- Model run: `privacy_v7_epoch1` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `label3`
- max_length: 2048  |  GPU: NVIDIA A10G  |  29.2s (8.6 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 160 | 64.0% |
| MEDIUM | 75 | 30.0% |
| HIGH | 15 | 6.0% |

Mean confidence: 0.784

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.6800**
- **Macro F1: 0.5551**  |  Weighted F1: 0.6609
- Macro precision: 0.6358  |  Macro recall: 0.5512
- Mean confidence when correct: 0.834  |  when wrong: 0.678

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.787 | 0.869 | 0.826 | 145 | 160 |
| MEDIUM | 0.453 | 0.557 | 0.500 | 61 | 75 |
| HIGH | 0.667 | 0.227 | 0.339 | 44 | 15 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 126 | 17 | 2 |
| **MEDIUM** | 24 | 34 | 3 |
| **HIGH** | 10 | 24 | 10 |
