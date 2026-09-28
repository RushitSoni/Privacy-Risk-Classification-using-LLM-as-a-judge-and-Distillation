"""
ModernBERT fine-tuning on Modal -- DYNAMIC training data.

Nothing about the dataset is hardcoded: point it at any CSV/TSV/Parquet/JSONL,
tell it which column is the text and which is the label, and it will
    * discover the label set from the data (LOW/MEDIUM/HIGH order is kept
      automatically when those are the labels, per VALID_LABELS in config.py),
    * split off a validation set (or use one you supply),
    * fine-tune ModernBERT with class-weighted cross-entropy,
    * save the best checkpoint + label map + metrics to the Modal data volume.

Uses the same Modal volumes as modal_app.py (DATA_VOLUME_NAME for data/models,
MODEL_CACHE_VOLUME_NAME so ModernBERT weights download only once).

USAGE
-----
    # Minimal (local file is uploaded to the volume automatically):
    modal run train_modernbert.py \
        --train-file data/final/final_train.csv \
        --text-col text --label-col label \
        --run-name privacy_v1

    # Any other dataset / column names -- same script:
    modal run train_modernbert.py \
        --train-file data/other/emails_labeled.csv \
        --text-col body --label-col risk \
        --val-file data/other/emails_val.csv

    # Already on the volume (skip the upload):
    modal run train_modernbert.py --train-file volume:/final/final_train.csv ...

    # Survives closing your laptop:
    modal run --detach train_modernbert.py --train-file ...

    # Bigger model / different GPU (GPU is chosen at import time, so use an env var):
    MODERNBERT_TRAIN_GPU=L40S modal run train_modernbert.py \
        --model-id answerdotai/ModernBERT-large --learning-rate 2e-5 --batch-size 8 ...

    # 2-minute smoke test on a slice:
    modal run train_modernbert.py --train-file ... --max-train-rows 300 --epochs 1

OUTPUT (on volume DATA_VOLUME_NAME)
-----------------------------------
    /models/<run-name>/            model weights + tokenizer (HF format)
    /models/<run-name>/label_map.json
    /models/<run-name>/train_config.json   every setting + data fingerprint
    /models/<run-name>/metrics.json        validation metrics + training log

    Pull it down:  modal volume get <DATA_VOLUME_NAME> /models/<run-name> models/<run-name>
"""

import json
import os
import time
from pathlib import Path

import modal

from config import (
    DATA_VOLUME_NAME,
    MODEL_CACHE_VOLUME_NAME,
    REMOTE_DATA_ROOT,
    VALID_LABELS,
)

# -------------------------------------------------------------------
# Settings that are not per-run (per-run settings are CLI flags below)
# -------------------------------------------------------------------

TRAIN_APP_NAME = "privacy-risk-modernbert-train"
DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"

# GPU is bound when the module is imported, so it is an env var, not a flag.
# ModernBERT-base fits comfortably on an A10G (24GB). Use L40S / A100 for -large
# or for very long max_length.
TRAIN_GPU = os.environ.get("MODERNBERT_TRAIN_GPU", "A10G")

HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(TRAIN_APP_NAME)

model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

HF_SECRET = modal.Secret.from_name("huggingface-secret")

# ModernBERT needs transformers >= 4.48.
train_image = (
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
# Small helpers
# -------------------------------------------------------------------

def _read_table(path):
    """Read csv / tsv / parquet / jsonl. CSV/TSV are read as raw strings so
    nothing (e.g. a label literally called 'NA') is silently turned into NaN."""
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


def _clean(df, text_col: str, label_col: str, name: str):
    """Keep only (text, label), stripped, with empty rows dropped."""
    for col in (text_col, label_col):
        if col not in df.columns:
            raise KeyError(
                f"[{name}] column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    n0 = len(out)

    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out["label"] = out["label"].fillna("").astype(str).str.strip()
    out = out[(out["text"] != "") & (out["label"] != "")].reset_index(drop=True)

    dropped = n0 - len(out)
    print(
        f"[{name}] {len(out)} usable rows"
        + (f" ({dropped} dropped: empty text/label)" if dropped else ""),
        flush=True,
    )
    return out


def _resolve_remote(root: Path, rel: str) -> Path:
    p = root / rel.lstrip("/")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found on volume '{DATA_VOLUME_NAME}'. "
            f"Check the path, or pass a local file and it will be uploaded."
        )
    return p


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -------------------------------------------------------------------
# GPU training function
# -------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        HF_CACHE_DIR: model_cache,
    },
    secrets=[HF_SECRET],
    timeout=60 * 60 * 6,
)
def train(cfg: dict) -> dict:
    import math

    import numpy as np
    import torch
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    t_start = time.time()
    root = Path(REMOTE_DATA_ROOT)
    run_name = cfg["run_name"]
    out_dir = root / "models" / run_name

    # Fail fast, before any expensive work.
    if out_dir.exists() and any(out_dir.iterdir()) and not cfg["overwrite"]:
        raise FileExistsError(
            f"Run '{run_name}' already exists on the volume at /models/{run_name}. "
            f"Choose a new --run-name or pass --overwrite yes."
        )

    set_seed(cfg["seed"])

    # ----------------------------------------------------------- data
    train_file = _resolve_remote(root, cfg["train_path"])
    train_df = _clean(_read_table(train_file), cfg["text_col"], cfg["label_col"], "train")

    if cfg["max_train_rows"] and cfg["max_train_rows"] < len(train_df):
        train_df = train_df.sample(n=cfg["max_train_rows"], random_state=cfg["seed"]).reset_index(drop=True)
        print(f"[train] using a random slice of {len(train_df)} rows (--max-train-rows)", flush=True)

    # Label set is discovered from the data. Keep LOW/MEDIUM/HIGH order if applicable.
    present = sorted(set(train_df["label"]))
    if set(present) <= set(VALID_LABELS):
        label_list = [l for l in VALID_LABELS if l in present]
    else:
        label_list = present

    if len(label_list) < 2:
        raise ValueError(f"Need at least 2 distinct labels, found: {label_list}")

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    if cfg["val_path"]:
        val_df = _clean(
            _read_table(_resolve_remote(root, cfg["val_path"])),
            cfg["text_col"], cfg["label_col"], "val",
        )
        unseen = ~val_df["label"].isin(label2id)
        if unseen.any():
            print(
                f"[val] WARNING: dropping {int(unseen.sum())} rows whose label is not in the "
                f"training set: {sorted(set(val_df.loc[unseen, 'label']))}",
                flush=True,
            )
            val_df = val_df[~unseen].reset_index(drop=True)
        split_mode = "provided_val_file"
    else:
        try:
            train_df, val_df = train_test_split(
                train_df, test_size=cfg["val_fraction"],
                stratify=train_df["label"], random_state=cfg["seed"],
            )
            split_mode = f"stratified_{cfg['val_fraction']}"
        except ValueError:
            print("[split] a class is too small to stratify; using a plain random split", flush=True)
            train_df, val_df = train_test_split(
                train_df, test_size=cfg["val_fraction"], random_state=cfg["seed"],
            )
            split_mode = f"random_{cfg['val_fraction']}"
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    if len(val_df) == 0:
        raise ValueError("Validation set is empty -- provide --val-file or raise --val-fraction.")

    train_y = train_df["label"].map(label2id).to_numpy()
    val_y = val_df["label"].map(label2id).to_numpy()

    train_counts = np.bincount(train_y, minlength=len(label_list))
    val_counts = np.bincount(val_y, minlength=len(label_list))

    print(f"[labels] {label_list}", flush=True)
    print(f"[train ] {len(train_df)} rows -> {dict(zip(label_list, train_counts.tolist()))}", flush=True)
    print(f"[val   ] {len(val_df)} rows -> {dict(zip(label_list, val_counts.tolist()))}", flush=True)

    # Class weights (pipeline: "Final training corpus ... class-weighted").
    if cfg["class_weights"] == "balanced":
        w = train_counts.sum() / (len(label_list) * np.maximum(train_counts, 1))
        class_weights = torch.tensor(w, dtype=torch.float)
        print(f"[loss  ] class-weighted CE: {dict(zip(label_list, np.round(w, 3).tolist()))}", flush=True)
    else:
        class_weights = None
        print("[loss  ] unweighted CE", flush=True)

    # ------------------------------------------------------ tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    max_length = cfg["max_length"]

    def _encode(texts):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    train_enc = _encode(train_df["text"])
    val_enc = _encode(val_df["text"])

    truncated = sum(len(ids) >= max_length for ids in train_enc["input_ids"])
    print(
        f"[tokens] {100 * truncated / len(train_df):.1f}% of train rows hit max_length={max_length}"
        f" (raise --max-length if that is high)",
        flush=True,
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": int(self.labels[i]),
            }

    train_ds = TextDataset(train_enc, train_y)
    val_ds = TextDataset(val_enc, val_y)

    # ---------------------------------------------------------- model
    config = AutoConfig.from_pretrained(
        cfg["model_id"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    # torch.compile on ModernBERT + variable-length batches causes constant recompiles.
    if hasattr(config, "reference_compile"):
        config.reference_compile = False

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], config=config)

    # -------------------------------------------------------- trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f, s = precision_recall_fscore_support(
            labels, preds, labels=list(range(len(label_list))), zero_division=0
        )
        present_cls = s > 0
        return {
            "accuracy": float((preds == labels).mean()),
            "f1_macro": float(f[present_cls].mean()),
            "f1_weighted": float((f * s).sum() / max(s.sum(), 1)),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = class_weights.to(logits.device) if class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    steps_per_epoch = math.ceil(len(train_ds) / (cfg["batch_size"] * cfg["grad_accum"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir="/tmp/ckpt",              # scratch disk; only the final model goes to the volume
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,         # keep the best epoch, not the last
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=use_bf16,
        logging_steps=25,
        report_to="none",
        seed=cfg["seed"],
        dataloader_num_workers=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(
        f"[train] {cfg['model_id']} | {total_steps} steps ({cfg['epochs']} epochs) | "
        f"bf16={use_bf16} | GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    trainer.train()

    # ------------------------------------------- final validation report
    pred = trainer.predict(val_ds)
    val_pred = np.argmax(pred.predictions, axis=-1)

    ids = list(range(len(label_list)))
    p, r, f, s = precision_recall_fscore_support(val_y, val_pred, labels=ids, zero_division=0)
    present_cls = s > 0
    cm = confusion_matrix(val_y, val_pred, labels=ids)

    metrics = {
        "accuracy": float((val_pred == val_y).mean()),
        "f1_macro": float(f[present_cls].mean()),
        "f1_weighted": float((f * s).sum() / max(s.sum(), 1)),
        "per_class": {
            label_list[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i]),
            }
            for i in ids
        },
        "confusion_matrix": {
            "labels": label_list,
            "rows_are": "ground_truth",
            "cols_are": "predicted",
            "matrix": cm.tolist(),
        },
    }

    print("\n=== VALIDATION (best checkpoint) ===", flush=True)
    print(f"accuracy={metrics['accuracy']:.4f}  macro-F1={metrics['f1_macro']:.4f}", flush=True)
    for l in label_list:
        m = metrics["per_class"][l]
        print(f"  {l:<10} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}", flush=True)

    # ------------------------------------------------------------ save
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))          # model weights (+ tokenizer via processing_class)
    tokenizer.save_pretrained(str(out_dir))

    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": label_list, "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()}}, indent=2)
    )

    train_config = {
        **{k: v for k, v in cfg.items()},
        "labels": label_list,
        "split_mode": split_mode,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "train_class_counts": dict(zip(label_list, train_counts.tolist())),
        "val_class_counts": dict(zip(label_list, val_counts.tolist())),
        "train_file_sha256": _sha256(train_file),
        "bf16": use_bf16,
        "gpu": torch.cuda.get_device_name(0),
        "train_seconds": round(time.time() - t_start, 1),
        "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

    (out_dir / "metrics.json").write_text(
        json.dumps({"validation": metrics, "log_history": trainer.state.log_history}, indent=2)
    )

    data_volume.commit()

    print(f"\nSaved -> /models/{run_name} on volume '{DATA_VOLUME_NAME}'", flush=True)
    return {
        "run_name": run_name,
        "model_dir": f"/models/{run_name}",
        "labels": label_list,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "train_seconds": train_config["train_seconds"],
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
    """Return the path (relative to the volume root) of the input file.
    'volume:/x/y.csv' is used as-is; anything else must be a local file and
    is uploaded to /<remote_subdir>/<filename> on the data volume."""
    if path_str.startswith("volume:"):
        return path_str[len("volume:"):]

    local = Path(path_str)
    if not local.is_file():
        raise FileNotFoundError(
            f"Local file not found: {local}\n"
            f"(If it is already on the Modal volume, prefix it with 'volume:', "
            f"e.g. volume:/final/final_train.csv)"
        )

    _check_columns_local(local, needed_cols)

    remote_rel = f"/{remote_subdir}/{local.name}"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local), remote_rel)
    print(f"Uploaded {local} -> volume '{DATA_VOLUME_NAME}':{remote_rel}", flush=True)
    return remote_rel


@app.local_entrypoint()
def main(
    train_file: str,
    text_col: str = "text",
    label_col: str = "label",
    val_file: str = "",
    run_name: str = "",
    model_id: str = DEFAULT_MODEL_ID,
    max_length: int = 512,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    val_fraction: float = 0.1,
    class_weights: str = "balanced",
    max_train_rows: int = 0,
    seed: int = 42,
    overwrite: str = "no",
):
    if class_weights not in ("balanced", "none"):
        raise ValueError("--class-weights must be 'balanced' or 'none'")

    run_name = run_name or f"modernbert_{time.strftime('%Y%m%d_%H%M%S')}"

    train_path = _stage_input(train_file, "train_inputs", [text_col, label_col])
    val_path = _stage_input(val_file, "train_inputs", [text_col, label_col]) if val_file else ""

    cfg = dict(
        train_path=train_path,
        val_path=val_path,
        text_col=text_col,
        label_col=label_col,
        run_name=run_name,
        model_id=model_id,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        val_fraction=val_fraction,
        class_weights=class_weights,
        max_train_rows=max_train_rows,
        seed=seed,
        overwrite=(overwrite.lower() in ("yes", "y", "true", "1")),
    )

    result = train.remote(cfg)

    print("\n================ DONE ================")
    print(json.dumps(result, indent=2))
    print(
        f"\nPull the model:\n"
        f"  modal volume get {DATA_VOLUME_NAME} /models/{run_name} models/{run_name}\n"
        f"Test it:\n"
        f"  modal run test_modernbert.py --run-name {run_name} "
        f"--test-file <file.csv> --text-col <text> --label-col <ground_truth>"
    )
