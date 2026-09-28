#!/usr/bin/env python3
"""
03_build_final_dataset.py

Builds a final, class-balanced training set for distillation by pooling
the {corpus}_majority2.csv files produced by 02_majority_vote.py across
any number of corpora.

ALLOCATION LOGIC (per class, e.g. HIGH / MEDIUM / LOW independently)
---------------------------------------------------------------------
Goal: hit `--per-class` samples per class while drawing from *every*
corpus that has data for that class (diversity), without letting one
huge corpus crowd out a tiny-but-valuable one.

  1. For each corpus, compute how many rows of this class it has
     available (its "availability").
  2. Any corpus whose availability is less than `--small-pool-frac`
     (default 20%) of the TOTAL availability across all corpora for
     this class is considered "scarce" for this class -> it contributes
     ALL of its available rows.
  3. The remaining target (per_class minus what scarce corpora gave)
     is split proportionally, by availability, among the remaining
     ("large") corpora -- capped by their own availability, with any
     leftover iteratively redistributed (waterfilling) so the target
     is still hit if a large corpus is capped.
  4. If total availability across ALL corpora for a class is less than
     `--per-class`, every corpus's full availability is taken (you get
     less than the target for that class -- a warning is printed).

Within each corpus's allocation for a class, unanimous (3/3) rows are
selected FIRST (highest label confidence), and only if that's not
enough are additional 2/3-majority rows randomly sampled to fill the
quota. This is deterministic given `--seed`.

Two files are written:
  - {out-prefix}_train.csv     -- the final balanced training set
  - {out-prefix}_holdout.csv   -- everything NOT selected, i.e. the
                                   natural (imbalanced) leftover pool,
                                   useful as a realistic held-out test
                                   set that mirrors real-world class
                                   distribution.

USAGE
-----
    python 03_build_final_dataset.py \
        --corpus nemotron=./out/nemotron_majority2.csv \
        --corpus wnut17=./out/wnut17_majority2.csv \
        --corpus enron=./out/enron_majority2.csv \
        --per-class 2000 \
        --small-pool-frac 0.20 \
        --seed 42 \
        --outdir ./out \
        --out-prefix final

Requires: pandas, numpy
"""

import argparse
import os

import numpy as np
import pandas as pd

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]


def parse_corpus_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--corpus must be in the form name=path.csv, got: {value}"
        )
    name, path = value.split("=", 1)
    return name, path


def allocate(target: int, availability: dict, small_pool_frac: float) -> dict:
    """
    availability: {corpus_name: n_available}
    Returns {corpus_name: n_to_take}, summing to min(target, total_available).
    """
    availability = {k: v for k, v in availability.items() if v > 0}
    total_avail = sum(availability.values())

    if total_avail <= target:
        # take everything, can't hit target -- caller should warn
        return dict(availability)

    # Step 1: scarce corpora take everything
    alloc = {}
    remaining_target = target
    large = {}
    for name, avail in availability.items():
        if avail < small_pool_frac * total_avail:
            alloc[name] = avail
            remaining_target -= avail
        else:
            large[name] = avail

    if not large:
        # everything was "scarce" (edge case) -- proportionally trim
        # (shouldn't normally happen given the small_pool_frac default)
        scale = target / total_avail
        return {k: int(round(v * scale)) for k, v in availability.items()}

    # Step 2: waterfilling proportional split among the large corpora
    remaining = dict(large)
    fixed = {}
    while remaining and remaining_target > 0:
        total_remaining_avail = sum(remaining.values())
        if total_remaining_avail <= remaining_target:
            fixed.update(remaining)
            remaining_target -= total_remaining_avail
            remaining = {}
            break
        overflow_found = False
        for name, avail in list(remaining.items()):
            share = remaining_target * avail / total_remaining_avail
            if share >= avail:
                fixed[name] = avail
                remaining_target -= avail
                del remaining[name]
                overflow_found = True
        if not overflow_found:
            # stable proportional split, no one exceeds their availability
            shares = {
                name: remaining_target * avail / sum(remaining.values())
                for name, avail in remaining.items()
            }
            # round while preserving the total
            floored = {k: int(np.floor(v)) for k, v in shares.items()}
            leftover = remaining_target - sum(floored.values())
            # hand out leftover units to the largest fractional remainders
            fracs = sorted(
                shares.items(), key=lambda kv: kv[1] - floored[kv[0]], reverse=True
            )
            for i in range(int(leftover)):
                name = fracs[i % len(fracs)][0]
                floored[name] += 1
            fixed.update(floored)
            remaining_target = 0
            remaining = {}
            break

    alloc.update(fixed)
    return alloc


def sample_unanimous_first(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    unan = df[df.unanimous]
    non_unan = df[~df.unanimous]
    if len(unan) >= n:
        idx = rng.choice(unan.index.values, size=n, replace=False)
        return df.loc[idx]
    take_non_unan = n - len(unan)
    if take_non_unan > len(non_unan):
        # shouldn't happen if allocation <= availability, but guard anyway
        take_non_unan = len(non_unan)
    idx_non_unan = rng.choice(non_unan.index.values, size=take_non_unan, replace=False)
    return pd.concat([unan, df.loc[idx_non_unan]])


def main():
    ap = argparse.ArgumentParser(description="Build final balanced distillation dataset")
    ap.add_argument(
        "--corpus", action="append", required=True, type=parse_corpus_arg,
        help="name=path.csv, repeatable. Each CSV must have columns: "
             "sample_id, text, label, unanimous (output of 02_majority_vote.py)",
    )
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--small-pool-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--out-prefix", default="final")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    corpora = {}
    for name, path in args.corpus:
        df = pd.read_csv(path)
        df["corpus"] = name
        corpora[name] = df

    all_selected = []
    all_leftover = []
    summary_rows = []

    for label in LABEL_ORDER:
        availability = {name: int((df.label == label).sum()) for name, df in corpora.items()}
        total_avail = sum(availability.values())

        if total_avail == 0:
            print(f"[WARN] No data available for class {label} in any corpus -- skipping.")
            continue
        if total_avail < args.per_class:
            print(
                f"[WARN] Only {total_avail} rows available for class {label} "
                f"(< target {args.per_class}). Taking all of them."
            )

        alloc = allocate(args.per_class, availability, args.small_pool_frac)

        for name, df in corpora.items():
            pool = df[df.label == label]
            n_take = alloc.get(name, 0)
            if n_take <= 0:
                continue
            chosen = sample_unanimous_first(pool, n_take, rng)
            all_selected.append(chosen)
            leftover = pool.drop(chosen.index)
            all_leftover.append(leftover)

            n_unan_chosen = int(chosen.unanimous.sum())
            summary_rows.append(
                {
                    "class": label,
                    "corpus": name,
                    "available": availability[name],
                    "unanimous_available": int(pool.unanimous.sum()),
                    "taken": n_take,
                    "taken_unanimous": n_unan_chosen,
                    "taken_majority_only": n_take - n_unan_chosen,
                }
            )

        # also carry forward any corpus/class combos that had 0 availability
        # (nothing to do, but keeps the summary complete)
        for name in corpora:
            if name not in alloc:
                summary_rows.append(
                    {
                        "class": label, "corpus": name,
                        "available": availability.get(name, 0),
                        "unanimous_available": 0, "taken": 0,
                        "taken_unanimous": 0, "taken_majority_only": 0,
                    }
                )

    train_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    holdout_df = pd.concat(all_leftover, ignore_index=True) if all_leftover else pd.DataFrame()

    # also add rows for classes/corpora that had 0 availability into holdout as N/A (skip)
    train_path = os.path.join(args.outdir, f"{args.out_prefix}_train.csv")
    holdout_path = os.path.join(args.outdir, f"{args.out_prefix}_holdout.csv")
    summary_path = os.path.join(args.outdir, f"{args.out_prefix}_allocation_summary.csv")

    keep_cols = ["sample_id", "text", "label", "unanimous", "corpus"]
    train_df[keep_cols].to_csv(train_path, index=False)
    holdout_df[keep_cols].to_csv(holdout_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*70}\nALLOCATION SUMMARY\n{'='*70}")
    print(summary_df.to_string(index=False))

    print(f"\nFinal train set: {len(train_df)} rows -> {train_path}")
    print(train_df.label.value_counts())
    print(f"\nHoldout (leftover, natural imbalance) set: {len(holdout_df)} rows -> {holdout_path}")
    print(holdout_df.label.value_counts())
    print(f"\nAllocation summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
