"""
Balance and truncate the majority-voted corpus to a fixed number of
samples per class, producing the final class-weighted training
corpus.

(pipeline image 1: "Balance and truncate: 3,000 samples per class"
-> "Final training corpus: 9,000 samples, class-weighted")
"""
import pandas as pd

from config import FINAL_DIR, SAMPLES_PER_CLASS, VALID_LABELS, RANDOM_SEED


def compute_class_weights(label_series: pd.Series) -> dict:
    """Inverse-frequency weights, for the (future) DistilBERT
    training stage's loss function."""
    counts = label_series.value_counts()
    n_classes = len(counts)
    total = counts.sum()
    return {label: total / (n_classes * c) for label, c in counts.items()}


def balance_and_truncate(voted_csv_paths: list, out_name: str = "final_training_corpus.csv") -> pd.DataFrame:
    """
    voted_csv_paths: list of per-corpus '*_majority.csv' paths
    (already majority-voted; no-majority rows already dropped
    upstream by voting.py). Combines all corpora, then samples down
    to SAMPLES_PER_CLASS per label. Does NOT pad/upsample a
    short class -- if a class has fewer rows than the target, all
    available rows for that class are kept and a warning is printed.
    """
    combined = pd.concat([pd.read_csv(p) for p in voted_csv_paths], ignore_index=True)
    combined = combined.rename(columns={"majority_label": "label"})

    balanced_parts = []
    for label in VALID_LABELS:
        pool = combined[combined["label"] == label]
        if len(pool) < SAMPLES_PER_CLASS:
            print(f"WARNING: class {label} has only {len(pool)} rows available "
                  f"(target was {SAMPLES_PER_CLASS}). Using all available rows "
                  f"instead of padding.")
            balanced_parts.append(pool)
        else:
            balanced_parts.append(pool.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED))

    final_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    class_weights = compute_class_weights(final_df["label"])

    out_path = FINAL_DIR / out_name
    final_df[["sample_id", "text", "label"]].to_csv(out_path, index=False)

    weights_path = FINAL_DIR / "class_weights.csv"
    pd.Series(class_weights, name="weight").rename_axis("label").to_csv(weights_path)

    print(f"Final corpus: {len(final_df)} rows -> {out_path}")
    print(f"Class counts:\n{final_df['label'].value_counts()}")
    print(f"Class weights (for DistilBERT loss) -> {weights_path}")

    return final_df
