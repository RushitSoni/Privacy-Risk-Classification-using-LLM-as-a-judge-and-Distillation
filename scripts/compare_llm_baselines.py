"""
Compare your ModernBERT student against the 13 LLM judges released with
"LLM-as-a-Judge for Privacy Evaluation?" (Meisenbacher et al., HAIPS'25),
on the same 250 texts, against the same label3 ground truth.

Usage:
    git clone https://github.com/sjmeis/privacy-judge.git
    python compare_llm_baselines.py \
        --student survey_with_label3_predictions.csv \
        --llm-file privacy-judge/data/llm_improved_combined.csv \
        --out-dir baseline_results

Notes on the released file
    * Rows = models. The first column holds the model name, and its header
      ("0") is misaligned by one: text i is in column str(i+1).
    * Values are integer 1-5 (5-run averages, rounded). NaN = no parseable answer.
    * Mapping to 3 classes matches your label3: 1-2 LOW, 3 MEDIUM, 4-5 HIGH.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABELS = ["LOW", "MEDIUM", "HIGH"]
L2I = {l: i for i, l in enumerate(LABELS)}
N_BOOT = 2000
RNG = np.random.default_rng(42)


def to3(r):
    """1-5 rating -> 0/1/2 (LOW/MEDIUM/HIGH); NaN stays -1."""
    out = np.full(len(r), -1, dtype=int)
    ok = ~np.isnan(r)
    rr = np.round(r[ok])
    out[ok] = np.where(rr <= 2, 0, np.where(rr == 3, 1, 2))
    return out


def metrics(y, p):
    ids = [0, 1, 2]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=ids, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float((y == p).mean()),
        "f1_macro": float(f1.mean()),
        "f1_LOW": float(f1[0]), "f1_MEDIUM": float(f1[1]), "f1_HIGH": float(f1[2]),
        "recall_LOW": float(rc[0]), "recall_MEDIUM": float(rc[1]), "recall_HIGH": float(rc[2]),
        "precision_HIGH": float(pr[2]),
        "pred_HIGH_count": int((p == 2).sum()),
        "cm": confusion_matrix(y, p, labels=ids).tolist(),
    }


def boot_ci(y, p, fn, n=N_BOOT):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        vals.append(fn(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def f1m(y, p):
    return precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)[2].mean()


def rec_high(y, p):
    m = y == 2
    return float((p[m] == 2).mean()) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--llm-file", required=True)
    ap.add_argument("--out-dir", default="baseline_results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(a.student).sort_values("text_id").reset_index(drop=True)
    assert len(u) == 250 and (u.text_id.values == np.arange(250)).all(), "expected text_id 0..249"
    y = u["label3"].map(L2I).to_numpy()
    human_mean = u["mean_rating"].to_numpy(float)

    d = pd.read_csv(a.llm_file)
    names = d.iloc[:, 0].tolist()
    ratings = d.iloc[:, 1:251].to_numpy(float)          # text i -> column i+1

    preds = {}
    # Student
    sp = u["predicted_label"].map(L2I).to_numpy()
    p3 = u[["prob_LOW", "prob_MEDIUM", "prob_HIGH"]].to_numpy()
    student_score = p3 @ np.array([1.0, 2.0, 3.0])       # expected level, for ranking metrics
    preds["STUDENT (ModernBERT)"] = (sp, student_score)
    # LLMs
    for n, r in zip(names, ratings):
        preds[n] = (to3(r), r)

    # Simple LLM ensemble: mean of the strong models (closed 7 + Llama-3.3-70B)
    strong = [n for n in names if any(k in n for k in
              ["claude", "gpt", "gemini", "70B"])]
    ens = np.nanmean(np.vstack([ratings[names.index(n)] for n in strong]), axis=0)
    preds["ENSEMBLE mean of 8 strong LLMs"] = (to3(ens), ens)

    # Majority-class baseline
    maj = np.full(250, np.bincount(y).argmax())
    preds["MAJORITY CLASS (always LOW)"] = (maj, maj.astype(float))

    rows, cms = [], {}
    for name, (p, score) in preds.items():
        m = p >= 0
        r = metrics(y[m], p[m])
        cms[name] = r.pop("cm")
        r["model"] = name
        r["coverage"] = float(m.mean())
        r["spearman_vs_human_mean"] = float(spearmanr(score[m], human_mean[m])[0]) \
            if np.nanstd(score[m]) > 0 else np.nan
        lo, hi = boot_ci(y[m], p[m], f1m)
        r["f1_macro_ci95_lo"], r["f1_macro_ci95_hi"] = lo, hi
        lo, hi = boot_ci(y[m], p[m], rec_high)
        r["recall_HIGH_ci95_lo"], r["recall_HIGH_ci95_hi"] = lo, hi
        rows.append(r)
    res = pd.DataFrame(rows).set_index("model")
    cols = ["n", "coverage", "accuracy", "f1_macro", "f1_macro_ci95_lo", "f1_macro_ci95_hi",
            "f1_LOW", "f1_MEDIUM", "f1_HIGH", "recall_LOW", "recall_MEDIUM", "recall_HIGH",
            "recall_HIGH_ci95_lo", "recall_HIGH_ci95_hi", "precision_HIGH", "pred_HIGH_count",
            "spearman_vs_human_mean"]
    res = res[cols].sort_values("f1_macro", ascending=False)
    res.round(4).to_csv(out / "comparison.csv")

    # Paired bootstrap: student minus each strong model, on rows the model parsed
    paired = []
    sp_arr = preds["STUDENT (ModernBERT)"][0]
    for n in strong + ["ENSEMBLE mean of 8 strong LLMs"]:
        p = preds[n][0]
        m = p >= 0
        yy, ss, pp = y[m], sp_arr[m], p[m]
        diffs = []
        for _ in range(N_BOOT):
            b = RNG.choice(len(yy), len(yy), replace=True)
            diffs.append(f1m(yy[b], ss[b]) - f1m(yy[b], pp[b]))
        paired.append({
            "vs": n,
            "student_f1_macro": f1m(yy, ss), "other_f1_macro": f1m(yy, pp),
            "diff": f1m(yy, ss) - f1m(yy, pp),
            "diff_ci95_lo": float(np.percentile(diffs, 2.5)),
            "diff_ci95_hi": float(np.percentile(diffs, 97.5)),
            "P(student better)": float(np.mean(np.array(diffs) > 0)),
        })
    pd.DataFrame(paired).round(4).to_csv(out / "paired_student_vs_llm.csv", index=False)

    # Per-source-dataset HIGH breakdown (ids 0-24 blog, 25-49 enron, ... 25 each)
    ds = ["BAC", "EE", "MQ", "MHB", "RC", "RLA", "RMHP", "TR", "TW", "YR"]
    # texts are grouped 25 per dataset in repo order; recover order from text_idx.json if present
    idxfile = Path(a.llm_file).parent / "text_idx.json"
    if idxfile.exists():
        ti = json.load(open(idxfile))
        src = np.empty(250, dtype=object)
        for k, ids_ in ti.items():
            src[ids_] = k
        bd = []
        for k in ti:
            m = src == k
            bd.append({
                "source": k, "n": int(m.sum()),
                "human_HIGH": int((y[m] == 2).sum()),
                "student_pred_HIGH": int((sp_arr[m] == 2).sum()),
                "student_HIGH_recall": rec_high(y[m], sp_arr[m]),
                "gpt4o_pred_HIGH": int((preds["gpt-4o"][0][m] == 2).sum()),
                "gpt4o_HIGH_recall": rec_high(y[m], preds["gpt-4o"][0][m]),
                "mean_human_rating": float(human_mean[m].mean()),
            })
        pd.DataFrame(bd).round(3).to_csv(out / "by_source_dataset.csv", index=False)

    with open(out / "confusion_matrices.json", "w") as f:
        json.dump({"labels": LABELS, "rows_are": "ground_truth", "cols_are": "predicted",
                   "matrices": cms}, f, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(res.round(3)[["coverage", "accuracy", "f1_macro", "f1_HIGH", "recall_HIGH",
                        "precision_HIGH", "pred_HIGH_count", "spearman_vs_human_mean"]])
    print("\nSaved to", out.resolve())


if __name__ == "__main__":
    main()
