# ModernBERT test summary: `survey_with_label3.csv`

- Model run: `privacy_v4` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `label3`
- max_length: 2048  |  GPU: NVIDIA A10  |  7.7s (32.3 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 155 | 62.0% |
| MEDIUM | 85 | 34.0% |
| HIGH | 10 | 4.0% |

Mean confidence: 0.955

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.7120**
- **Macro F1: 0.5726**  |  Weighted F1: 0.6880
- Macro precision: 0.7049  |  Macro recall: 0.5812
- Mean confidence when correct: 0.970  |  when wrong: 0.917

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.832 | 0.890 | 0.860 | 145 | 155 |
| MEDIUM | 0.482 | 0.672 | 0.562 | 61 | 85 |
| HIGH | 0.800 | 0.182 | 0.296 | 44 | 10 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 129 | 16 | 0 |
| **MEDIUM** | 18 | 41 | 2 |
| **HIGH** | 8 | 28 | 8 |
