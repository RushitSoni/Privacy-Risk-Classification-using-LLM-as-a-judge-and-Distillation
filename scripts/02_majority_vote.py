#!/usr/bin/env python3
"""
02_majority_vote.py

Builds two labeled CSVs per corpus from the three raw judge CSVs:

  1. {corpus}_majority2.csv  -- every row that has a 2-of-3 (or 3-of-3)
                                 majority label. Columns include a
                                 boolean `unanimous` flag (True if all
                                 3 judges agreed) so downstream scripts
                                 can prioritize the highest-confidence
                                 rows first. Rows where all 3 judges gave
                                 different labels (no majority) are DROPPED
                                 from this file and written separately to
                                 {corpus}_no_consensus.csv for inspection.

  2. {corpus}_majority3.csv  -- only the strict subset where all 3 judges
                                 unanimously agree (3-of-3). This is a
                                 pure subset of majority2.csv.

A judge label of "UNKNOWN" (failed parse) is treated as an abstention:
the majority is computed over whichever judges returned a valid
LOW/MEDIUM/HIGH label. If fewer than 2 valid labels remain, or the
valid labels disagree with no majority, the row goes to no_consensus.

USAGE
-----
    python 02_majority_vote.py \
        --judge-a path/to/xyz_judge_a.csv \
        --judge-b path/to/xyz_judge_b.csv \
        --judge-c path/to/xyz_judge_c.csv \
        --corpus-name xyz \
        --outdir ./out

Requires: pandas
"""

import argparse
import os

import pandas as pd

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]


def find_label_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith("_label")]
    if len(cols) != 1:
        raise ValueError(f"Expected exactly one *_label column, found {cols}")
    return cols[0]


def find_parse_status_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith("_parse_status")]
    if len(cols) != 1:
        raise ValueError(f"Expected exactly one *_parse_status column, found {cols}")
    return cols[0]


def load_judge(path: str, generic_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = find_label_col(df)
    status_col = find_parse_status_col(df)
    df = df.rename(
        columns={label_col: f"{generic_name}_label", status_col: f"{generic_name}_parse_status"}
    )
    return df[["sample_id", "text", f"{generic_name}_label", f"{generic_name}_parse_status"]]


def majority_row(row) -> tuple:
    """Return (majority_label_or_None, unanimous_bool)."""
    labels = [row.judge_a_label, row.judge_b_label, row.judge_c_label]
    valid = [l for l in labels if l in LABEL_ORDER]
    if len(valid) < 2:
        return None, False
    counts = pd.Series(valid).value_counts()
    top_count = counts.iloc[0]
    if top_count < 2:
        return None, False
    top_label = counts.index[0]
    unanimous = len(valid) == 3 and counts.iloc[0] == 3
    return top_label, unanimous


def main():
    ap = argparse.ArgumentParser(description="Build 2-vote and 3-vote majority CSVs")
    ap.add_argument("--judge-a", required=True)
    ap.add_argument("--judge-b", required=True)
    ap.add_argument("--judge-c", required=True)
    ap.add_argument("--corpus-name", default=None)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    corpus_name = args.corpus_name or os.path.basename(args.judge_a).split("_judge_a")[0]
    os.makedirs(args.outdir, exist_ok=True)

    a = load_judge(args.judge_a, "judge_a")
    b = load_judge(args.judge_b, "judge_b")
    c = load_judge(args.judge_c, "judge_c")

    assert set(a.sample_id) == set(b.sample_id) == set(c.sample_id), "sample_id sets differ!"
    m = a.merge(b[["sample_id", "judge_b_label", "judge_b_parse_status"]], on="sample_id")
    m = m.merge(c[["sample_id", "judge_c_label", "judge_c_parse_status"]], on="sample_id")

    results = m.apply(majority_row, axis=1, result_type="expand")
    m["label"] = results[0]
    m["unanimous"] = results[1]

    no_consensus = m[m.label.isna()].copy()
    majority2 = m[m.label.notna()].copy()
    majority3 = majority2[majority2.unanimous].copy()

    keep_cols = [
        "sample_id", "text", "label", "unanimous",
        "judge_a_label", "judge_b_label", "judge_c_label",
    ]
    majority2_out = majority2[keep_cols]
    majority3_out = majority3[keep_cols]

    m2_path = os.path.join(args.outdir, f"{corpus_name}_majority2.csv")
    m3_path = os.path.join(args.outdir, f"{corpus_name}_majority3.csv")
    nc_path = os.path.join(args.outdir, f"{corpus_name}_no_consensus.csv")

    majority2_out.to_csv(m2_path, index=False)
    majority3_out.to_csv(m3_path, index=False)
    no_consensus[keep_cols[:-1] + ["judge_a_label", "judge_b_label", "judge_c_label"]].to_csv(
        nc_path, index=False
    )

    print(f"\n=== {corpus_name} ===")
    print(f"Total rows:            {len(m)}")
    print(f"2/3+ majority (kept):  {len(majority2_out)}  -> {m2_path}")
    print(f"  of which unanimous:  {len(majority3_out)}  -> {m3_path}")
    print(f"No consensus (dropped): {len(no_consensus)}  -> {nc_path}")
    print("\nmajority2 label distribution:")
    print(majority2_out.label.value_counts())
    print("\nmajority3 (unanimous) label distribution:")
    print(majority3_out.label.value_counts())


if __name__ == "__main__":
    main()
