"""
End-to-end orchestrator. Run stages individually (recommended, so
you can inspect output between steps) or all at once.

    python pipeline.py --stage source
    python pipeline.py --stage judge --judge judge_a
    python pipeline.py --stage judge --judge judge_b
    python pipeline.py --stage judge --judge judge_c
    python pipeline.py --stage vote
    python pipeline.py --stage balance
    python pipeline.py --stage train_student_stub
    python pipeline.py --stage all

Each stage reads the previous stage's output from disk (config.py's
*_DIR paths) -- nothing is held in memory across stages, so you can
run this over multiple days/sessions, exactly like the original
notebook's session-gated design, and re-run any single stage without
re-running the others.
"""
import argparse

from config import RAW_FILES, VOTED_DIR, JUDGED_DIR
from sourcing import build_sourced_corpus
from judges_runner import run_single_judge
from voting import combine_judged_csvs
from balance import balance_and_truncate


def stage_source():
    for corpus_name in RAW_FILES:
        build_sourced_corpus(corpus_name)


def stage_judge(judge_key: str):
    for corpus_name in RAW_FILES:
        run_single_judge(judge_key, corpus_name)


def stage_vote():
    voted_paths = []
    for corpus_name in RAW_FILES:
        judge_csv_paths = {
            "judge_a": JUDGED_DIR / f"{corpus_name}_judge_a.csv",
            "judge_b": JUDGED_DIR / f"{corpus_name}_judge_b.csv",
            "judge_c": JUDGED_DIR / f"{corpus_name}_judge_c.csv",
        }
        missing = [str(p) for p in judge_csv_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing judge outputs for '{corpus_name}': {missing}. "
                f"Run --stage judge for all three judges on this corpus first."
            )
        combine_judged_csvs(judge_csv_paths, corpus_name)
        voted_paths.append(str(VOTED_DIR / f"{corpus_name}_majority.csv"))
    return voted_paths


def stage_balance():
    voted_paths = [str(VOTED_DIR / f"{c}_majority.csv") for c in RAW_FILES]
    missing = [p for p in voted_paths if not __import__("pathlib").Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing voted outputs: {missing}. Run --stage vote first.")
    balance_and_truncate(voted_paths)


def stage_train_student_stub():
    print(
        "Student model training (DistilBERT) is the NEXT stage and is "
        "NOT implemented here -- see data/final/final_training_corpus.csv "
        "and data/final/class_weights.csv as its input contract."
    )


def main():
    parser = argparse.ArgumentParser(description="Privacy-risk labeling pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["source", "judge", "vote", "balance", "train_student_stub", "all"],
    )
    parser.add_argument(
        "--judge",
        choices=["judge_a", "judge_b", "judge_c"],
        help="Required when --stage judge (run one judge at a time, same "
             "one-model-per-session pattern as the original notebook).",
    )
    args = parser.parse_args()

    if args.stage == "source":
        stage_source()
    elif args.stage == "judge":
        if not args.judge:
            parser.error("--stage judge requires --judge {judge_a,judge_b,judge_c}")
        stage_judge(args.judge)
    elif args.stage == "vote":
        stage_vote()
    elif args.stage == "balance":
        stage_balance()
    elif args.stage == "train_student_stub":
        stage_train_student_stub()
    elif args.stage == "all":
        stage_source()
        for jk in ("judge_a", "judge_b", "judge_c"):
            stage_judge(jk)
        stage_vote()
        stage_balance()
        stage_train_student_stub()


if __name__ == "__main__":
    main()
