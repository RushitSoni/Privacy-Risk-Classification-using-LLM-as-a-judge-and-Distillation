#!/usr/bin/env python3
"""
run_jev_eval.py
===============
Zero-shot evaluation of TypeSafe JEV as a privacy-risk judge (LOW / MEDIUM / HIGH)
on the 250-sample human-annotated survey set, using YOUR prompt (prompts.py).

What it does
------------
1. Loads survey_with_label3.csv (gold = `label3`).
2. Converts prompts.SYSTEM_PROMPT into a JEV `choice` question
   (rubric -> `instructions`, LOW/MEDIUM/HIGH -> `criteria`; the "Return ONLY ...
   Label:" output-format block is dropped because JEV returns probabilities, not text).
3. Calls POST https://api.typesafe.ai/v1/systemone once per sample (concurrent,
   with retry/backoff, resumable).
4. Saves everything needed for later review (see "Outputs").
5. Computes accuracy / macro-F1 / weighted-F1 / per-class P-R-F1 / confusion matrix,
   and a confidence analysis (accuracy vs q, AUROC, confident errors).

Setup
-----
    pip install requests pandas numpy scikit-learn
    export TYPESAFE_API_KEY="..."            # Windows PowerShell: $env:TYPESAFE_API_KEY="..."
    # put this file next to prompts.py and survey_with_label3.csv

Usage
-----
    python run_jev_eval.py --dry-run                     # inspect the request, no API call, no key needed
    python run_jev_eval.py --limit 5 --out-dir jev_results/smoke   # 5-sample smoke test
    python run_jev_eval.py                               # full run (250 samples)
    python run_jev_eval.py                               # re-running resumes; only failed/missing ids are retried
    python run_jev_eval.py --report-only                 # recompute metrics/report from saved raw responses

Outputs (in --out-dir, default jev_results/jev-1.13.0)
------------------------------------------------------
    raw_responses.jsonl   one JSON line per API call: full answer, usage, latency, attempts, errors
    predictions.csv       per-sample gold, pred, probabilities, q, correctness (with text)
    errors.csv            misclassified samples, most-confident errors first
    metrics.json          all metrics (machine-readable)
    report.md             human-readable report
    prompt_used.txt       exact instructions + criteria sent to JEV
    run_config.json       args, model requested/reported, hashes, versions, timing, token usage/cost

Notes
-----
* Run ONCE with a fixed prompt and report. Editing the prompt after seeing results on
  these 250 samples inflates the score.
* Model is pinned to `jev-1.13.0` (the version in the JEV paper). The alias `jev-latest`
  can move to a newer model without notice.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABELS = ["LOW", "MEDIUM", "HIGH"]
QUESTION_ID = "privacy_risk"          # not sent to the model (per TypeSafe docs)
PRICE_PER_MTOK = 0.042                # USD per million input tokens for jev-1.13.0 (per docs; verify)
PROB_SUM_TOL = 0.025                  # same tolerance the JEV paper used for normalization

# Definitions copied verbatim from prompts.py (the "- LOW: / - MEDIUM: / - HIGH:" lines).
CRITERIA = {
    "LOW": "little to no realistic harm from disclosure.",
    "MEDIUM": "disclosure could cause meaningful harm (e.g. embarrassment, moderate fraud or contact risk).",
    "HIGH": "disclosure could cause severe harm (e.g. identity theft, financial loss, safety risk, legal exposure).",
}


# --------------------------------------------------------------------------- prompt
def build_instructions() -> str:
    """Turn prompts.SYSTEM_PROMPT into JEV `instructions` (drop the text-output format block)."""
    try:
        from prompts import SYSTEM_PROMPT
    except ImportError:
        sys.exit("ERROR: cannot import prompts.py. Put run_jev_eval.py in the same folder as prompts.py.")

    marker = "Return ONLY in this format"
    if marker not in SYSTEM_PROMPT:
        print("WARNING: output-format marker not found in prompts.py; sending the full prompt unchanged.",
              file=sys.stderr)
        body = SYSTEM_PROMPT.strip()
    else:
        body = SYSTEM_PROMPT.split(marker)[0].strip()

    # The state carries the text; the original prompt said "the text below".
    body = body.replace("classify the text below", "classify the text given as the state")

    return body


def build_payload(text: str, model: str, instructions: str) -> dict:
    return {
        "state": text,
        "model": model,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": instructions,
                "criteria": CRITERIA,
            }
        },
    }


# --------------------------------------------------------------------------- API
class FatalAPIError(Exception):
    """Errors that make continuing pointless (bad key, forbidden)."""


_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def call_api(url: str, key: str, payload: dict, max_retries: int, timeout: float):
    """POST with exponential backoff on 429/529/5xx/connection errors.
    Returns (json_or_None, attempts, error_or_None)."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        retry_after = None
        try:
            r = _session().post(url, json=payload, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                try:
                    return r.json(), attempt, None
                except ValueError:
                    last_err = "HTTP 200 but body was not valid JSON"
            elif r.status_code in (401, 403):
                raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:300]} (check TYPESAFE_API_KEY)")
            elif r.status_code == 422:
                return None, attempt, f"HTTP 422 (request rejected): {r.text[:500]}"
            elif r.status_code in (429, 529) or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass
            else:
                return None, attempt, f"HTTP {r.status_code}: {r.text[:500]}"
        if attempt < max_retries:
            backoff = min(60.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            time.sleep(max(backoff, retry_after or 0))
    return None, max_retries, last_err


def parse_answer(data: dict):
    """Validate a /v1/systemone response. Returns (parsed_dict_or_None, error_or_None)."""
    try:
        ans = data["answers"][QUESTION_ID]
        choice = ans["choice"]
        probs = {k: float(v) for k, v in ans["probabilities"].items()}
    except (KeyError, TypeError, ValueError) as e:
        return None, f"malformed response: {type(e).__name__}: {e}"
    if choice not in LABELS:
        return None, f"choice {choice!r} not in {LABELS}"
    if set(probs) != set(LABELS):
        return None, f"probability keys {sorted(probs)} != {LABELS}"
    if not all(np.isfinite(list(probs.values()))):
        return None, "non-finite probability"
    if abs(sum(probs.values()) - 1.0) > PROB_SUM_TOL:
        return None, f"probabilities sum to {sum(probs.values()):.4f}"
    argmax = max(probs, key=probs.get)
    if probs[choice] < probs[argmax] - 1e-9:
        return None, f"choice {choice} is not the argmax ({argmax})"
    return {
        "choice": choice,
        "probs": probs,
        "q": max(probs.values()),
        "native_confidence": ans.get("confidence"),
    }, None


# --------------------------------------------------------------------------- run
def load_latest_records(raw_path: Path) -> dict:
    """Latest record per text_id from the JSONL log (later lines override earlier ones)."""
    recs = {}
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[int(r["text_id"])] = r
    return recs


def run_calls(df, args, instructions, raw_path: Path, key: str):
    recs = load_latest_records(raw_path)
    todo = [(int(r.text_id), r.text) for r in df.itertuples() if not recs.get(int(r.text_id), {}).get("ok")]
    print(f"{len(df)} samples | already done: {len(df) - len(todo)} | to call: {len(todo)}")
    if not todo:
        return

    url = args.base_url.rstrip("/") + "/v1/systemone"
    lock = threading.Lock()
    counter = {"n": 0}
    t_start = time.time()
    fout = open(raw_path, "a", encoding="utf-8")

    def work(tid: int, text: str):
        payload = build_payload(text, args.model, instructions)
        t0 = time.time()
        data, attempts, err = call_api(url, key, payload, args.max_retries, args.timeout)
        latency = time.time() - t0
        rec = {
            "text_id": tid,
            "ts": datetime.now(timezone.utc).isoformat(),
            "model_requested": args.model,
            "model_reported": None,
            "ok": False,
            "error": err,
            "attempts": attempts,
            "latency_s": round(latency, 4),
            "usage": None,
            "parsed": None,
            "raw_answer": None,
        }
        if data is not None:
            rec["model_reported"] = data.get("model")
            rec["usage"] = data.get("usage")
            rec["raw_answer"] = data.get("answers", {}).get(QUESTION_ID)
            parsed, perr = parse_answer(data)
            rec["parsed"], rec["error"] = parsed, perr
            rec["ok"] = parsed is not None
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(todo):
                print(f"  {counter['n']}/{len(todo)} done ({time.time() - t_start:.0f}s)")
        return rec

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, tid, text) for tid, text in todo]
            try:
                for f in as_completed(futs):
                    f.result()
            except FatalAPIError as e:
                ex.shutdown(wait=False, cancel_futures=True)
                fout.close()
                sys.exit(f"FATAL: {e}")
    finally:
        if not fout.closed:
            fout.close()


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def q_bin(q):
    if q >= 1 - 1e-9:
        return "=1"
    for lo, hi, name in [(0, .6, "<0.6"), (.6, .7, "[0.6,0.7)"), (.7, .8, "[0.7,0.8)"), (.8, .9, "[0.8,0.9)"),
                         (.9, .95, "[0.9,0.95)"), (.95, .99, "[0.95,0.99)"), (.99, 1.0, "[0.99,1)")]:
        if lo <= q < hi:
            return name
    return "?"


BIN_ORDER = ["<0.6", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,0.95)", "[0.95,0.99)", "[0.99,1)", "=1"]


def confidence_analysis(pred_df):
    valid = pred_df[pred_df["status"] == "ok"].copy()
    res = {"n_valid": int(len(valid))}
    if len(valid) == 0:
        return res
    valid["is_error"] = (~valid["correct"]).astype(int)
    res["mean_q_correct"] = float(valid.loc[valid["correct"], "q"].mean()) if valid["correct"].any() else None
    res["mean_q_incorrect"] = float(valid.loc[~valid["correct"], "q"].mean()) if (~valid["correct"]).any() else None
    # AUROC: errors are positives, scored by 1-q (same convention as the JEV paper)
    if valid["is_error"].nunique() == 2:
        res["error_detection_auroc"] = float(roc_auc_score(valid["is_error"], 1 - valid["q"]))
    else:
        res["error_detection_auroc"] = None

    # threshold table (DESCRIPTIVE / post hoc -- do not use to tune and then report on the same 250)
    rows = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        acc_m = valid["q"] >= tau
        rows.append({
            "tau": tau,
            "accepted_n": int(acc_m.sum()),
            "coverage": float(acc_m.mean()),
            "acc_accepted": float(valid.loc[acc_m, "correct"].mean()) if acc_m.any() else None,
            "acc_escalated": float(valid.loc[~acc_m, "correct"].mean()) if (~acc_m).any() else None,
        })
    res["threshold_table"] = rows

    valid["q_bin"] = valid["q"].map(q_bin)
    b = []
    for name in BIN_ORDER:
        s = valid[valid["q_bin"] == name]
        if len(s):
            b.append({"bin": name, "n": int(len(s)), "accuracy": float(s["correct"].mean())})
    res["accuracy_by_q_bin"] = b

    conf_err = valid[(valid["q"] >= 0.9) & (~valid["correct"])]
    res["confident_errors_q>=0.9"] = {
        "n": int(len(conf_err)),
        "of_n_confident": int((valid["q"] >= 0.9).sum()),
        "by_gold_to_pred": {f"{g}->{p}": int(c) for (g, p), c in conf_err.groupby(["gold", "pred"]).size().items()},
    }
    return res


def build_predictions(df, recs):
    rows = []
    for r in df.itertuples():
        tid = int(r.text_id)
        rec = recs.get(tid)
        ok = bool(rec and rec.get("ok"))
        parsed = rec["parsed"] if ok else None
        pred = parsed["choice"] if ok else "INVALID"
        rows.append({
            "text_id": tid,
            "gold": r.gold,
            "pred": pred,
            "correct": pred == r.gold,
            "status": "ok" if ok else ("failed" if rec else "not_run"),
            "q": parsed["q"] if ok else np.nan,
            "native_confidence": parsed["native_confidence"] if ok else np.nan,
            "p_LOW": parsed["probs"]["LOW"] if ok else np.nan,
            "p_MEDIUM": parsed["probs"]["MEDIUM"] if ok else np.nan,
            "p_HIGH": parsed["probs"]["HIGH"] if ok else np.nan,
            "input_tokens": (rec.get("usage") or {}).get("input_tokens") if rec else None,
            "latency_s": rec.get("latency_s") if rec else None,
            "attempts": rec.get("attempts") if rec else None,
            "error": rec.get("error") if rec else "not run",
            "mean_rating": r.mean_rating,
            "num_ratings": r.num_ratings,
            "text": r.text,
        })
    return pd.DataFrame(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def make_report(m, cfg) -> str:
    L = []
    L.append(f"# JEV zero-shot privacy-risk evaluation\n")
    L.append(f"- Model requested: `{cfg['model_requested']}`; reported by API: `{', '.join(cfg['models_reported']) or 'n/a'}`")
    L.append(f"- Samples: {m['n_total']} (valid outputs: {m['n_valid']}, invalid/failed: {m['n_invalid']}; invalid counted as errors)")
    cc = m["gold_class_counts"]
    L.append(f"- Gold class counts in this CSV: LOW {cc['LOW']}, MEDIUM {cc['MEDIUM']}, HIGH {cc['HIGH']}")
    L.append(f"- Prompt hash: `{cfg['instructions_sha256'][:16]}`; CSV hash: `{cfg['csv_sha256'][:16]}`")
    if m["n_total"] != 250:
        L.append(f"\n> WARNING: only {m['n_total']} samples evaluated (smoke test / --limit). Do not report these numbers.")
    L.append("\n## Overall\n")
    L.append(md_table(["Metric", "Value"], [
        [k, f3(m["overall"][k])]
        for k in ["accuracy", "macro_f1", "weighted_f1"]]))
    L.append("\n## Per class\n")
    L.append(md_table(["Class", "Precision", "Recall", "F1", "Support"],
                      [[c, f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), v["support"]]
                       for c, v in m["per_class"].items()]))
    L.append("\n## Confusion matrix (rows = human gold, cols = JEV)\n")
    cm = m["confusion_matrix"]
    L.append(md_table(["gold \\ pred"] + LABELS + ["INVALID"],
                      [[g] + [cm[g][p] for p in LABELS] + [cm[g]["INVALID"]] for g in LABELS]))

    c = m["confidence"]
    L.append("\n## Does JEV's confidence track correctness?\n")
    L.append(f"- Mean q when correct: {f3(c.get('mean_q_correct'))}; when incorrect: {f3(c.get('mean_q_incorrect'))}")
    L.append(f"- Error-detection AUROC (errors positive, score 1-q): {f3(c.get('error_detection_auroc'))} "
             f"(0.5 = no signal)")
    if c.get("accuracy_by_q_bin"):
        L.append("\n" + md_table(["q bin", "n", "accuracy"],
                                 [[b["bin"], b["n"], f3(b["accuracy"])] for b in c["accuracy_by_q_bin"]]))
        L.append("\n**Threshold table (descriptive, post hoc):**\n")
        L.append(md_table(["tau", "accepted n", "coverage", "acc if accepted", "acc if escalated"],
                          [[t["tau"], t["accepted_n"], f3(t["coverage"]), f3(t["acc_accepted"]), f3(t["acc_escalated"])]
                           for t in c["threshold_table"]]))
        ce = c["confident_errors_q>=0.9"]
        L.append(f"\nConfident errors (q>=0.9): {ce['n']} of {ce['of_n_confident']} confident predictions. "
                 f"Breakdown: {ce['by_gold_to_pred'] or 'none'}")
    L.append("\n## Cost / latency\n")
    L.append(f"- Input tokens: {cfg['total_input_tokens']:,}; est. cost at ${PRICE_PER_MTOK}/Mtok: "
             f"${cfg['estimated_cost_usd']:.4f} (output tokens are free per TypeSafe docs; verify current pricing)")
    L.append(f"- Median per-call latency: {f3(cfg['median_latency_s'])} s (concurrent client, includes retries)")
    L.append("\n## Interpretation checklist\n")
    L.append("- AUROC near 0.5 or confident errors concentrated in MEDIUM? Confidence gating will not help here.\n"
             "- Do NOT tune the prompt or a threshold on these 250 and then report them; split off a dev subset first.")
    return "\n".join(L)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/test/survey_with_label3.csv")
    ap.add_argument("--model", default="jev-1.13.0", help="pinned version (alias 'jev-latest' can change)")
    ap.add_argument("--out-dir", default=None, help="default: jev_results/<model>")
    ap.add_argument("--base-url", default="https://api.typesafe.ai")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (limit is 1,200 req/min)")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="print first request and exit; no API call")
    ap.add_argument("--report-only", action="store_true", help="skip API calls; rebuild outputs from raw_responses.jsonl")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir or f"jev_results/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"text_id", "text", "label3"}
    if not need <= set(df.columns):
        sys.exit(f"ERROR: CSV must contain columns {sorted(need)}; found {list(df.columns)}")
    df["gold"] = df["label3"].astype(str).str.strip().str.upper()
    bad = set(df["gold"]) - set(LABELS)
    if bad:
        sys.exit(f"ERROR: unexpected gold labels in label3: {bad}")
    df["text"] = df["text"].fillna("").astype(str)
    if (df["text"].str.strip() == "").any():
        print(f"WARNING: {(df['text'].str.strip() == '').sum()} empty texts (API may reject them; counted as invalid)")
    for col in ("mean_rating", "num_ratings"):
        if col not in df.columns:
            df[col] = np.nan
    if df["text_id"].duplicated().any():
        sys.exit("ERROR: duplicate text_id values")
    if args.limit:
        df = df.head(args.limit).copy()

    instructions = build_instructions()
    (out / "prompt_used.txt").write_text(
        "=== instructions ===\n" + instructions + "\n\n=== criteria ===\n" + json.dumps(CRITERIA, indent=2) + "\n",
        encoding="utf-8")

    if args.dry_run:
        first = df.iloc[0]
        print("Request that would be sent (first sample):\n")
        print(f"POST {args.base_url.rstrip('/')}/v1/systemone")
        print(json.dumps(build_payload(first["text"], args.model, instructions), indent=2, ensure_ascii=False))
        print(f"\nInstructions length: {len(instructions)} chars. Also written to {out / 'prompt_used.txt'}")
        return

    raw_path = out / "raw_responses.jsonl"
    t_run = time.time()
    if not args.report_only:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            sys.exit("ERROR: set TYPESAFE_API_KEY (get one at https://console.typesafe.ai/keys). "
                     "Use --dry-run to inspect the request without a key.")
        run_calls(df, args, instructions, raw_path, key)
    wall = time.time() - t_run

    recs = load_latest_records(raw_path)
    pred_df = build_predictions(df, recs)
    pred_df.to_csv(out / "predictions.csv", index=False)
    (pred_df[~pred_df["correct"]].sort_values("q", ascending=False, na_position="last")
     .to_csv(out / "errors.csv", index=False))

    y_true, y_pred = pred_df["gold"].tolist(), pred_df["pred"].tolist()
    overall = compute_metrics(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i, c in enumerate(LABELS)}
    cm_labels = LABELS + ["INVALID"]
    cm_arr = confusion_matrix(y_true, y_pred, labels=cm_labels)
    cm = {g: {pp: int(cm_arr[i][j]) for j, pp in enumerate(cm_labels)} for i, g in enumerate(LABELS)}
    gold_counts = {c: int((pred_df["gold"] == c).sum()) for c in LABELS}

    metrics = {
        "n_total": int(len(pred_df)),
        "n_valid": int((pred_df["status"] == "ok").sum()),
        "n_invalid": int((pred_df["status"] != "ok").sum()),
        "gold_class_counts": gold_counts,
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confidence": confidence_analysis(pred_df),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    toks = [r_["usage"]["input_tokens"] for r_ in recs.values() if r_.get("usage") and r_["usage"].get("input_tokens")]
    lat = [r_["latency_s"] for r_ in recs.values() if r_.get("latency_s") is not None]
    cfg = {
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "model_requested": args.model,
        "models_reported": sorted({r_["model_reported"] for r_ in recs.values() if r_.get("model_reported")}),
        "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "csv_sha256": sha256_file(args.csv),
        "n_texts": int(len(df)),
        "total_input_tokens": int(sum(toks)),
        "estimated_cost_usd": sum(toks) * PRICE_PER_MTOK / 1e6,
        "median_latency_s": float(np.median(lat)) if lat else None,
        "wall_seconds_this_invocation": round(wall, 1),
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "requests": requests.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    report = make_report(metrics, cfg)
    (out / "report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"JEV ({', '.join(cfg['models_reported']) or args.model}) on {len(df)} samples "
          f"(valid {metrics['n_valid']}, invalid {metrics['n_invalid']})")
    for k in ["accuracy", "macro_f1", "weighted_f1"]:
        print(f"  {k:12s} {overall[k]:.3f}")
    print(f"  recall LOW/MED/HIGH: {per_class['LOW']['recall']:.2f} / {per_class['MEDIUM']['recall']:.2f} / "
          f"{per_class['HIGH']['recall']:.2f}")
    print(f"  error-detection AUROC of q: {f3(metrics['confidence'].get('error_detection_auroc'))}")
    print(f"\nAll outputs saved in: {out.resolve()}")
    if metrics["n_invalid"]:
        print(f"NOTE: {metrics['n_invalid']} samples failed; re-run the same command to retry only those.")


if __name__ == "__main__":
    main()
