# ModernBERT test summary: `survey_with_label3.csv`

- Model run: `privacy_v7_epoch10` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `label3`
- max_length: 2048  |  GPU: NVIDIA A10  |  11.3s (22.2 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 170 | 68.0% |
| MEDIUM | 73 | 29.2% |
| HIGH | 7 | 2.8% |

Mean confidence: 0.978

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.6880**
- **Macro F1: 0.5379**  |  Weighted F1: 0.6545
- Macro precision: 0.7454  |  Macro recall: 0.5400
- Mean confidence when correct: 0.987  |  when wrong: 0.960

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.771 | 0.903 | 0.832 | 145 | 170 |
| MEDIUM | 0.466 | 0.557 | 0.507 | 61 | 73 |
| HIGH | 1.000 | 0.159 | 0.275 | 44 | 7 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 131 | 14 | 0 |
| **MEDIUM** | 27 | 34 | 0 |
| **HIGH** | 12 | 25 | 7 |
