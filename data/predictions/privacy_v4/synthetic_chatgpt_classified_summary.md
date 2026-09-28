# ModernBERT test summary: `synthetic_chatgpt_classified.csv`

- Model run: `privacy_v4` (answerdotai/ModernBERT-base)
- Rows in file: 250  |  scored: 250  |  empty text (skipped): 0
- Text column: `text`  |  ground-truth column: `privacy_leak_level`
- max_length: 2048  |  GPU: NVIDIA A10  |  3.6s (69.7 rows/s)

## Predicted label distribution

| label | count | share |
|---|---:|---:|
| LOW | 86 | 34.4% |
| MEDIUM | 98 | 39.2% |
| HIGH | 66 | 26.4% |

Mean confidence: 0.974

## Evaluation vs ground truth

- Rows evaluated: 250
- **Accuracy: 0.8840**
- **Macro F1: 0.8839**  |  Weighted F1: 0.8842
- Macro precision: 0.8957  |  Macro recall: 0.8837
- Mean confidence when correct: 0.984  |  when wrong: 0.899

### Per class

| class | precision | recall | F1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| LOW | 0.942 | 0.964 | 0.953 | 84 | 86 |
| MEDIUM | 0.776 | 0.916 | 0.840 | 83 | 98 |
| HIGH | 0.970 | 0.771 | 0.859 | 83 | 66 |

### Confusion matrix (rows = ground truth, columns = predicted)

| | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| **LOW** | 81 | 3 | 0 |
| **MEDIUM** | 5 | 76 | 2 |
| **HIGH** | 0 | 19 | 64 |
