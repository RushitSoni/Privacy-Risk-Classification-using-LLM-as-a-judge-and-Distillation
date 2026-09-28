"""
Local-side driver for running each judge one at a time (mirrors the
original notebook's "one dataset/session at a time" pattern, but
here it's one JUDGE at a time across a sourced corpus). Call this
from pipeline.py, or invoke it directly.

This loop runs ON YOUR MACHINE: each chunk is a blocking remote call
to the GPU judge (which runs on Modal), but the loop driving those
calls is a local process. If it's killed for any reason -- closed
laptop, sleep, network drop, Ctrl-C -- the loop stops. See
run_judge_job in modal_app.py (invoked via `modal run --detach`) for
a version of this same logic that survives that.

What THIS version gives you instead is resumability: progress is
checkpointed to disk after every chunk, so re-running the same
command picks up where it left off rather than re-labeling (and
re-paying GPU time for) rows already done.
"""
import pandas as pd

from config import SOURCED_DIR, JUDGED_DIR, JUDGES
from parsing import parse_label
from checkpoint_io import load_checkpoint, append_checkpoint
from modal_app import app, JUDGE_CLASSES


def run_single_judge(judge_key: str, corpus_name: str, chunk_size: int = 256, resume: bool = True):
    """
    Runs ONE judge over ONE already-sourced corpus CSV, calling the
    matching Modal GPU class remotely, and writes a labeled CSV to
    data/judged/{corpus_name}_{judge_key}.csv.

    Resumable: a checkpoint at
    data/judged/checkpoints/{corpus_name}_{judge_key}.jsonl records
    every labeled row as it completes. On start, rows already in that
    file are skipped -- only the remaining rows get sent to the GPU.
    Pass resume=False, or delete the checkpoint file, to force a
    clean re-run from scratch.
    """
    assert judge_key in JUDGES, f"Unknown judge: {judge_key}"
    src_path = SOURCED_DIR / f"{corpus_name}_sourced.csv"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Missing {src_path} -- run `python pipeline.py --stage source` first."
        )
    df = pd.read_csv(src_path)

    ckpt_path = JUDGED_DIR / "checkpoints" / f"{corpus_name}_{judge_key}.jsonl"
    done = load_checkpoint(ckpt_path) if resume else {}
    if done:
        print(f"[{judge_key}/{corpus_name}] resuming: {len(done)}/{len(df)} rows already labeled")

    pending = df[~df["sample_id"].isin(done.keys())]
    texts = pending["text"].astype(str).tolist()
    ids = pending["sample_id"].tolist()

    if texts:
        with app.run():
            # Instantiated once: every chunk below lands on the same
            # warm container, so weights load only once for this run.
            judge = JUDGE_CLASSES[judge_key]()
            for start in range(0, len(texts), chunk_size):
                chunk_ids = ids[start:start + chunk_size]
                chunk_texts = texts[start:start + chunk_size]
                raw_outputs = judge.generate.remote(chunk_texts)

                new_rows = []
                for sid, raw in zip(chunk_ids, raw_outputs):
                    label, status = parse_label(raw)
                    new_rows.append({"sample_id": sid, "label": label, "status": status})

                append_checkpoint(ckpt_path, new_rows)   # on disk before moving on
                for r in new_rows:
                    done[r["sample_id"]] = r

                print(f"[{judge_key}/{corpus_name}] {len(done)}/{len(df)} done")
    else:
        print(f"[{judge_key}/{corpus_name}] nothing to do -- all {len(df)} rows already checkpointed")

    # Assemble the final CSV from the FULL checkpoint (prior run(s) +
    # this one), in the sourced file's original row order.
    df[f"{judge_key}_label"] = df["sample_id"].map(lambda s: done[s]["label"])
    df[f"{judge_key}_parse_status"] = df["sample_id"].map(lambda s: done[s]["status"])

    n_unknown = sum(1 for r in done.values() if r["status"] == "fallback_unknown")
    if n_unknown:
        print(f"[{judge_key}/{corpus_name}] WARNING: {n_unknown} rows fell back to UNKNOWN")

    out_path = JUDGED_DIR / f"{corpus_name}_{judge_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return out_path
