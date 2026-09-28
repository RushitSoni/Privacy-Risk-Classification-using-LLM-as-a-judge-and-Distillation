"""
Shared incremental-checkpoint helpers for the judging stage. Used by
BOTH orchestrators -- the local loop in judges_runner.py and the
detached-on-Modal function in modal_app.py::run_judge_job -- so a run
started one way and resumed the other reads/writes the exact same
checkpoint format.

Format: one JSON object per line (.jsonl), appended and flushed after
every chunk. That matters more than it looks: a crash mid-write can
only ever cost you the last unflushed line, never the rows already on
disk, and an append-only file is safe to write to repeatedly without
ever re-reading + rewriting the whole thing.

    {"sample_id": "...", "label": "HIGH", "status": "ok"}
"""
import json
from pathlib import Path


def load_checkpoint(path: Path) -> dict:
    """
    Returns {sample_id: {"label": ..., "status": ...}} for rows
    already labeled in a prior (possibly interrupted) run.
    Missing file -> empty dict, not an error -- that's just a fresh run.
    """
    done = {}
    if not path.exists():
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Last line of a crash mid-write. Drop it silently --
                # that row will simply get re-labeled this run.
                continue
            done[row["sample_id"]] = {"label": row["label"], "status": row["status"]}
    return done


def append_checkpoint(path: Path, rows: list) -> None:
    """Append-only write, flushed before returning -- rows already on
    disk are never touched, so this is safe to call after every chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
