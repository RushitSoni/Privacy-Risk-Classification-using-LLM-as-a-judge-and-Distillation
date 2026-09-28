"""
ModernBERT testing / inference on Modal -- DYNAMIC test data.

Give it ANY test file, the text column, and (optionally) the ground-truth column.
It loads a model trained by train_modernbert.py and writes:

    <name>_predictions.csv   your original file + predicted_label, predicted_confidence,
                             prob_<LABEL> per class, is_correct (if ground truth given)
    <name>_summary.json      accuracy, macro/weighted F1, per-class P/R/F1, confusion
                             matrix, label distributions, confidence stats, run metadata
    <name>_summary.md        the same summary, human-readable

If --label-col is omitted the script just predicts (no metrics).

USAGE
-----
    modal run test_modernbert.py \
        --run-name privacy_v1 \
        --test-file data/other/test_set.csv \
        --text-col body \
        --label-col true_risk

    # Test file already on the volume:
    modal run test_modernbert.py --run-name privacy_v1 \
        --test-file volume:/train_inputs/test_set.csv --text-col text --label-col label

    # Predict only, no ground truth:
    modal run test_modernbert.py --run-name privacy_v1 --test-file new_data.csv --text-col text

    # Different GPU (bound at import time, so env var):
    MODERNBERT_TEST_GPU=L4 modal run test_modernbert.py ...

OUTPUT
------
    Local:   data/predictions/<run-name>/   (or --out-dir)
    Volume:  /predictions/<run-name>/       on DATA_VOLUME_NAME
"""

import json
import os
import time
from pathlib import Path

import modal

from config import (
    DATA_DIR,
    DATA_VOLUME_NAME,
    MODEL_CACHE_VOLUME_NAME,
    REMOTE_DATA_ROOT,
)

TEST_APP_NAME = "privacy-risk-modernbert-test"

# Inference is light; A10G is plenty for ModernBERT-base. Use L4 to save money.
TEST_GPU = os.environ.get("MODERNBERT_TEST_GPU", "A10G")

HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(TEST_APP_NAME)

model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

HF_SECRET = modal.Secret.from_name("huggingface-secret")

test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.48.0",
        "accelerate",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "numpy",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": HF_CACHE_DIR,
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_python_source("config")
)


# -------------------------------------------------------------------
# Pure helpers (no GPU needed; unit-testable)
# -------------------------------------------------------------------

def _read_table(path):
    """Read csv / tsv / parquet / jsonl. CSV/TSV are read as raw strings so the
    output file keeps your original values byte-for-byte (no 007 -> 7, no NA -> NaN)."""
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)

    raise ValueError(
        f"Unsupported file type '{suffix}' for {p.name}. "
        f"Use .csv, .tsv, .parquet, .jsonl or .json"
    )


def _unique_col(name: str, existing) -> str:
    """Never overwrite one of the user's own columns."""
    if name not in existing:
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


def _normalize_ground_truth(gt_series, labels):
    """Map raw ground-truth values onto the model's label set (strip + case-insensitive).
    Returns (idx_array with -1 where unmappable, n_missing, {unseen_value: count})."""
    import numpy as np

    lower_map = {l.lower(): i for i, l in enumerate(labels)}

    idx = np.full(len(gt_series), -1, dtype=int)
    n_missing = 0
    unseen = {}

    for pos, raw in enumerate(gt_series.tolist()):
        value = "" if raw is None else str(raw).strip()
        if value == "" or value.lower() == "nan":
            n_missing += 1
            continue
        hit = lower_map.get(value.lower())
        if hit is None:
            unseen[value] = unseen.get(value, 0) + 1
        else:
            idx[pos] = hit

    return idx, n_missing, unseen


def _compute_metrics(y_true, y_pred, labels):
    """Accuracy, macro/weighted P/R/F1 (macro = over classes present in the ground truth),
    per-class table and confusion matrix."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    ids = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    present = s > 0
    total = max(int(s.sum()), 1)
    cm = confusion_matrix(y_true, y_pred, labels=ids)

    return {
        "n_evaluated": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(p[present].mean()),
        "recall_macro": float(r[present].mean()),
        "f1_macro": float(f[present].mean()),
        "f1_weighted": float((f * s).sum() / total),
        "macro_averaged_over": [labels[i] for i in ids if present[i]],
        "per_class": {
            labels[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
                "predicted_count": int((y_pred == i).sum()),
            }
            for i in ids
        },
        "confusion_matrix": {
            "labels": labels,
            "rows_are": "ground_truth",
            "cols_are": "predicted",
            "matrix": cm.tolist(),
        },
    }


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _render_summary_md(s: dict) -> str:
    m = s["meta"]
    lines = [
        f"# ModernBERT test summary: `{m['test_file']}`",
        "",
        f"- Model run: `{m['run_name']}` ({m.get('model_id', '?')})",
        f"- Rows in file: {m['n_rows']}  |  scored: {m['n_scored']}  |  empty text (skipped): {m['n_empty_text']}",
        f"- Text column: `{m['text_col']}`  |  ground-truth column: `{m['label_col'] or '(none)'}`",
        f"- max_length: {m['max_length']}  |  GPU: {m['gpu']}  |  {m['seconds']}s ({m['rows_per_sec']} rows/s)",
        "",
        "## Predicted label distribution",
        "",
        "| label | count | share |",
        "|---|---:|---:|",
    ]
    for l, c in s["predicted_distribution"].items():
        share = c / max(m["n_scored"], 1)
        lines.append(f"| {l} | {c} | {share:.1%} |")

    lines += [
        "",
        f"Mean confidence: {s['confidence']['mean']:.3f}",
    ]

    ev = s.get("evaluation")
    if ev is None:
        lines += ["", "_No ground-truth column supplied, so no accuracy metrics were computed._"]
        return "\n".join(lines) + "\n"

    gt = s["ground_truth_quality"]
    lines += [
        "",
        "## Evaluation vs ground truth",
        "",
        f"- Rows evaluated: {ev['n_evaluated']}",
        f"- **Accuracy: {ev['accuracy']:.4f}**",
        f"- **Macro F1: {ev['f1_macro']:.4f}**  |  Weighted F1: {ev['f1_weighted']:.4f}",
        f"- Macro precision: {ev['precision_macro']:.4f}  |  Macro recall: {ev['recall_macro']:.4f}",
        f"- Mean confidence when correct: {_fmt(s['confidence'].get('mean_when_correct'))}"
        f"  |  when wrong: {_fmt(s['confidence'].get('mean_when_wrong'))}",
    ]
    if gt["n_missing_ground_truth"] or gt["unseen_ground_truth_values"]:
        lines += [
            "",
            f"> Not evaluated: {gt['n_missing_ground_truth']} rows with blank ground truth, "
            f"{sum(gt['unseen_ground_truth_values'].values())} rows with a label the model never saw "
            f"({gt['unseen_ground_truth_values'] or 'none'}).",
        ]

    lines += ["", "### Per class", "", "| class | precision | recall | F1 | support | predicted |", "|---|---:|---:|---:|---:|---:|"]
    for l, v in ev["per_class"].items():
        lines.append(
            f"| {l} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} | {v['predicted_count']} |"
        )

    cm = ev["confusion_matrix"]
    lines += ["", "### Confusion matrix (rows = ground truth, columns = predicted)", ""]
    lines.append("| | " + " | ".join(cm["labels"]) + " |")
    lines.append("|---|" + "---:|" * len(cm["labels"]))
    for l, row in zip(cm["labels"], cm["matrix"]):
        lines.append(f"| **{l}** | " + " | ".join(str(x) for x in row) + " |")

    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------
# GPU prediction function
# -------------------------------------------------------------------

@app.function(
    image=test_image,
    gpu=TEST_GPU,
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        HF_CACHE_DIR: model_cache,
    },
    secrets=[HF_SECRET],
    timeout=60 * 60 * 3,
)
def predict(cfg: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t_start = time.time()
    root = Path(REMOTE_DATA_ROOT)
    run_name = cfg["run_name"]
    model_dir = root / "models" / run_name

    # ------------------------------------------------------ load model
    if not model_dir.exists():
        available = sorted(p.name for p in (root / "models").glob("*")) if (root / "models").exists() else []
        raise FileNotFoundError(
            f"No trained model '{run_name}' at /models/{run_name} on volume '{DATA_VOLUME_NAME}'. "
            f"Available runs: {available or 'none'}"
        )

    label_map = json.loads((model_dir / "label_map.json").read_text())
    labels = label_map["labels"]

    train_cfg, train_metrics = {}, {}
    if (model_dir / "train_config.json").exists():
        train_cfg = json.loads((model_dir / "train_config.json").read_text())
    if (model_dir / "metrics.json").exists():
        train_metrics = json.loads((model_dir / "metrics.json").read_text())

    max_length = cfg["max_length"] or train_cfg.get("max_length", 512)

    # ------------------------------------------------------- load data
    test_file = root / cfg["test_path"].lstrip("/")
    if not test_file.exists():
        raise FileNotFoundError(f"{test_file} not found on volume '{DATA_VOLUME_NAME}'.")

    df = _read_table(test_file)
    text_col, label_col = cfg["text_col"], cfg["label_col"]

    for col in (text_col, label_col):
        if col and col not in df.columns:
            raise KeyError(f"Column '{col}' not found in test file. Available columns: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).str.strip()
    valid = (texts != "").to_numpy()
    n_rows, n_valid = len(df), int(valid.sum())
    print(f"[test] {n_rows} rows, {n_valid} with non-empty text", flush=True)

    # ------------------------------------------------------- inference
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to("cuda").eval()
    use_bf16 = torch.cuda.is_bf16_supported()

    valid_pos = np.flatnonzero(valid)
    lengths = texts.iloc[valid_pos].str.len().to_numpy()
    order = valid_pos[np.argsort(-lengths, kind="stable")]     # longest first: OOM shows up immediately, padding stays low

    probs = np.zeros((n_rows, len(labels)), dtype=np.float32)
    bs = cfg["batch_size"]
    n_batches = (len(order) + bs - 1) // bs

    with torch.inference_mode():
        for b in range(n_batches):
            pos = order[b * bs:(b + 1) * bs]
            enc = tokenizer(
                texts.iloc[pos].tolist(),
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to("cuda") for k, v in enc.items() if k in ("input_ids", "attention_mask")}

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits

            probs[pos] = torch.softmax(logits.float(), dim=-1).cpu().numpy()

            if (b + 1) % 10 == 0 or b + 1 == n_batches:
                done = min((b + 1) * bs, len(order))
                print(f"[test] {done}/{len(order)} rows predicted", flush=True)

    pred_idx = np.where(valid, probs.argmax(axis=1), -1)
    confidence = np.where(valid, probs.max(axis=1), np.nan)

    # ----------------------------------------------- build output file
    out = df.copy()
    c_pred = _unique_col("predicted_label", out.columns)
    out[c_pred] = [labels[i] if i >= 0 else "" for i in pred_idx]

    c_conf = _unique_col("predicted_confidence", out.columns)
    out[c_conf] = np.where(valid, np.round(confidence, 4), np.nan)

    for j, l in enumerate(labels):
        c = _unique_col(f"prob_{l}", out.columns)
        out[c] = np.where(valid, np.round(probs[:, j], 4), np.nan)

    c_status = _unique_col("prediction_status", out.columns)
    out[c_status] = np.where(valid, "ok", "empty_text")

    # ------------------------------------------------------- summary
    scored_pred = pred_idx[valid]
    pred_dist = {l: int((scored_pred == i).sum()) for i, l in enumerate(labels)}

    summary = {
        "meta": {
            "run_name": run_name,
            "model_id": train_cfg.get("model_id"),
            "test_file": test_file.name,
            "text_col": text_col,
            "label_col": label_col,
            "n_rows": n_rows,
            "n_scored": n_valid,
            "n_empty_text": n_rows - n_valid,
            "max_length": int(max_length),
            "batch_size": bs,
            "gpu": torch.cuda.get_device_name(0),
            "labels": labels,
            "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        },
        "model_training_reference": {
            "n_train": train_cfg.get("n_train"),
            "train_file": train_cfg.get("train_path"),
            "validation_accuracy": train_metrics.get("validation", {}).get("accuracy"),
            "validation_f1_macro": train_metrics.get("validation", {}).get("f1_macro"),
        },
        "predicted_distribution": pred_dist,
        "confidence": {"mean": float(np.nanmean(confidence)) if n_valid else float("nan")},
        "evaluation": None,
        "ground_truth_quality": None,
    }

    if label_col:
        gt_idx, n_missing, unseen = _normalize_ground_truth(df[label_col], labels)
        eval_mask = valid & (gt_idx >= 0)

        summary["ground_truth_distribution"] = {
            l: int((gt_idx[valid] == i).sum()) for i, l in enumerate(labels)
        }
        summary["ground_truth_quality"] = {
            "n_missing_ground_truth": int(n_missing),
            "unseen_ground_truth_values": dict(sorted(unseen.items(), key=lambda kv: -kv[1])[:20]),
        }

        c_correct = _unique_col("is_correct", out.columns)
        correct_col = np.full(n_rows, None, dtype=object)
        correct_col[eval_mask] = (pred_idx[eval_mask] == gt_idx[eval_mask])
        out[c_correct] = correct_col

        if eval_mask.any():
            summary["evaluation"] = _compute_metrics(gt_idx[eval_mask], pred_idx[eval_mask], labels)
            ok = pred_idx[eval_mask] == gt_idx[eval_mask]
            conf_eval = confidence[eval_mask]
            summary["confidence"]["mean_when_correct"] = float(conf_eval[ok].mean()) if ok.any() else None
            summary["confidence"]["mean_when_wrong"] = float(conf_eval[~ok].mean()) if (~ok).any() else None
        else:
            print("[test] WARNING: no rows had a usable ground-truth label; metrics skipped.", flush=True)

    elapsed = time.time() - t_start
    summary["meta"]["seconds"] = round(elapsed, 1)
    summary["meta"]["rows_per_sec"] = round(n_valid / max(elapsed, 1e-9), 1)

    # ------------------------------------------------------- write
    stem = cfg["output_name"] or Path(test_file.name).stem
    out_rel = f"predictions/{run_name}"
    out_dir = root / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / f"{stem}_predictions.csv"
    out.to_csv(pred_path, index=False)

    summary_md = _render_summary_md(summary)
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{stem}_summary.md").write_text(summary_md)

    data_volume.commit()

    print(f"[test] saved /{out_rel}/{stem}_predictions.csv (+ summary) on volume '{DATA_VOLUME_NAME}'", flush=True)

    return {
        "predictions_remote_path": f"/{out_rel}/{pred_path.name}",
        "stem": stem,
        "summary": summary,
        "summary_md": summary_md,
    }


# -------------------------------------------------------------------
# Local entrypoint (runs on your machine)
# -------------------------------------------------------------------

def _check_columns_local(path: Path, needed) -> None:
    """Fail in one second on a column typo instead of after a GPU cold start."""
    import csv

    suffix = path.suffix.lower()
    if suffix not in (".csv", ".tsv"):
        return
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter="\t" if suffix == ".tsv" else ","), [])
    missing = [c for c in needed if c and c not in header]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in {path.name}. Available: {header}")


def _stage_input(path_str: str, remote_subdir: str, needed_cols) -> str:
    """'volume:/x/y.csv' is used as-is; anything else must be a local file and is
    uploaded to /<remote_subdir>/<filename> on the data volume."""
    if path_str.startswith("volume:"):
        return path_str[len("volume:"):]

    local = Path(path_str)
    if not local.is_file():
        raise FileNotFoundError(
            f"Local file not found: {local}\n"
            f"(If it is already on the Modal volume, prefix it with 'volume:', "
            f"e.g. volume:/train_inputs/test_set.csv)"
        )

    _check_columns_local(local, needed_cols)

    remote_rel = f"/{remote_subdir}/{local.name}"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local), remote_rel)
    print(f"Uploaded {local} -> volume '{DATA_VOLUME_NAME}':{remote_rel}", flush=True)
    return remote_rel


@app.local_entrypoint()
def main(
    test_file: str,
    run_name: str,
    text_col: str = "text",
    label_col: str = "",
    batch_size: int = 64,
    max_length: int = 0,
    output_name: str = "",
    out_dir: str = "",
):
    test_path = _stage_input(test_file, "test_inputs", [text_col, label_col])

    cfg = dict(
        test_path=test_path,
        run_name=run_name,
        text_col=text_col,
        label_col=label_col,
        batch_size=batch_size,
        max_length=max_length,          # 0 = reuse the max_length the model was trained with
        output_name=output_name,
    )

    result = predict.remote(cfg)

    local_dir = Path(out_dir) if out_dir else DATA_DIR / "predictions" / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    stem = result["stem"]

    # Summary files come back in the return value -- no download step to fail.
    (local_dir / f"{stem}_summary.json").write_text(json.dumps(result["summary"], indent=2))
    (local_dir / f"{stem}_summary.md").write_text(result["summary_md"])

    # The predictions file can be large, so stream it from the volume.
    pred_local = local_dir / f"{stem}_predictions.csv"
    remote_path = result["predictions_remote_path"]
    try:
        with open(pred_local, "wb") as f:
            for chunk in data_volume.read_file(remote_path):
                f.write(chunk)
    except Exception as e:  # noqa: BLE001
        print(
            f"\n(Could not auto-download predictions: {e})\n"
            f"Pull it manually:\n"
            f"  modal volume get {DATA_VOLUME_NAME} {remote_path} {pred_local}"
        )

    print("\n" + result["summary_md"])
    print(f"Saved locally in {local_dir}/:")
    print(f"  {stem}_predictions.csv")
    print(f"  {stem}_summary.json")
    print(f"  {stem}_summary.md")
