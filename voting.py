"""
Combines the three judges' per-sample labels into one majority-vote
label, computes Fleiss' kappa for inter-judge reliability, and logs
+ drops any sample without a majority.

(pipeline image 1: "Majority vote: 3 judges + Fleiss' kappa"
-> "No majority: sample dropped, logged")
"""
from collections import Counter

import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from config import VOTED_DIR, MIN_AGREEING_JUDGES, VALID_LABELS


def majority_vote(row_labels: list):
    """Returns (label, agreeing_count), or (None, 0) if no label
    reaches MIN_AGREEING_JUDGES votes."""
    counts = Counter(l for l in row_labels if l in VALID_LABELS)
    if not counts:
        return None, 0
    label, n = counts.most_common(1)[0]
    if n >= MIN_AGREEING_JUDGES:
        return label, n
    return None, 0


def combine_judged_csvs(judge_csv_paths: dict, corpus_name: str):
    """
    judge_csv_paths: {"judge_a": path, "judge_b": path, "judge_c": path},
    all for the SAME corpus and same underlying sourced rows.

    Returns (kept_df, dropped_df, fleiss_kappa_value).
    """
    dfs = {k: pd.read_csv(p) for k, p in judge_csv_paths.items()}

    base = dfs["judge_a"][["sample_id", "text"]].copy()
    label_cols = []
    for jk, df in dfs.items():
        col = f"{jk}_label"
        base[col] = df[col].values
        label_cols.append(col)

    votes, agree_counts = [], []
    for _, row in base.iterrows():
        label, n = majority_vote([row[c] for c in label_cols])
        votes.append(label)
        agree_counts.append(n)

    base["majority_label"] = votes
    base["n_agreeing"] = agree_counts

    kept = base[base["majority_label"].notna()].copy()
    dropped = base[base["majority_label"].isna()].copy()

    # Fleiss' kappa over rows where all three judges produced a
    # valid (non-UNKNOWN) label -- mixing in UNKNOWN as its own
    # "category" would distort agreement, so it's excluded here.
    valid_mask = base[label_cols].isin(VALID_LABELS).all(axis=1)
    kappa = None
    if valid_mask.sum() >= 2:
        rating_table, _ = aggregate_raters(base.loc[valid_mask, label_cols].values)
        kappa = fleiss_kappa(rating_table)

    kept_path = VOTED_DIR / f"{corpus_name}_majority.csv"
    dropped_path = VOTED_DIR / f"{corpus_name}_no_majority_dropped.csv"
    kept.to_csv(kept_path, index=False)
    dropped.to_csv(dropped_path, index=False)

    kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a (too few valid rows)"
    print(f"[{corpus_name}] kept={len(kept)} dropped={len(dropped)} "
          f"Fleiss' kappa={kappa_str}  -> {kept_path}")

    return kept, dropped, kappa
