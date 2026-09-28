# ModernBERT test summary: `synthetic_gemini_classified.csv`

- Model run: `privacy_v4` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `privacy_leak_level`
- max_length: 2048  |  GPU: NVIDIA A10  |  3.7s (67.5 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 134 | 53.6% |
| MEDIUM | 61 | 24.4% |
| HIGH | 55 | 22.0% |

Mean confidence: 0.977

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.7960**
- **Macro F1: 0.7439**  |  Weighted F1: 0.8120
- Macro precision: 0.7588  |  Macro recall: 0.7538
- Mean confidence when correct: 0.986  |  when wrong: 0.942

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.903 | 0.852 | 0.877 | 142 | 134 |
| MEDIUM | 0.410 | 0.641 | 0.500 | 39 | 61 |
| HIGH | 0.964 | 0.768 | 0.855 | 69 | 55 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 121 | 20 | 1 |
| **MEDIUM** | 13 | 25 | 1 |
| **HIGH** | 0 | 16 | 53 |
