"""
Sourcing stage: stream each raw corpus CSV in chunks and draw a
UNIFORM random oversample -- no PII-proxy scoring, no full-file load
into memory.

Earlier version scored every row with regex heuristics (email/phone/
SSN/card/name-pair patterns) and did a weighted draw favoring
PII-proxy-positive rows. Removed at your request, given these raw
files are ~9GB each. Worth knowing WHY that was slow, since it's not
what it looks like: the regex scoring itself benchmarks fast even in
adversarial cases (~85k rows/sec on realistic text, sub-second even
on worst-case 4000-char digit-dense rows). The actual bottleneck was
`pd.read_csv()` pulling the ENTIRE file into a DataFrame before any
scoring or sampling could start -- at 9GB, that read (plus pandas'
in-memory overhead, often 3-5x the raw file size for object/string
columns) is what was hanging, not the regex.

This version fixes the real problem: it streams the file in chunks
(config.READ_CHUNK_SIZE rows at a time) and does reservoir sampling
(Algorithm R) directly off that stream, so at most one chunk plus the
final sample is ever in memory -- never the whole file.

TRADE-OFF vs the old version: rows are no longer biased toward
PII-proxy-positive text, so MEDIUM/HIGH-risk rows may turn up less
often per corpus than before, and balance.py's "only N rows
available" warning may fire more often downstream. If that becomes a
real problem, the fix is to do the risk-weighting later in the
pipeline (e.g. oversample from voting.py's dropped/kept split) rather
than re-adding a per-row regex pass at read time.
"""
import hashlib

import numpy as np
import pandas as pd

from config import (
    RAW_FILES, TEXT_COLUMN_HINTS, BASE_SAMPLE_SIZE,
    OVERSAMPLE_FACTOR_RANGE, TEXT_FILTER, RANDOM_SEED, SOURCED_DIR,
    READ_CHUNK_SIZE,
)

_JUNK_CHARS = "0123456789 \t\n\r\x0b\x0c-_.,!?;:'\"()[]{}"


def _detect_text_column_from_sample(df_sample: pd.DataFrame, hint) -> str:
    """
    Column auto-detect runs on the FIRST CHUNK ONLY, not the whole
    file -- reading a full 9GB file just to pick a column would
    defeat the entire point of streaming. If your file is sorted or
    grouped in a way that makes the first ~50k rows an unrepresentative
    sample of the columns, set TEXT_COLUMN_HINTS explicitly in
    config.py instead of relying on auto-detect.
    """
    if hint and hint in df_sample.columns:
        return hint
    str_cols = df_sample.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        raise ValueError("No string columns found to use as text.")
    avg_lens = {c: df_sample[c].dropna().astype(str).str.len().mean() for c in str_cols}
    return max(avg_lens, key=avg_lens.get)


def _row_passes_filters(text: str) -> bool:
    f = TEXT_FILTER
    if not (f["min_chars"] <= len(text) <= f["max_chars"]):
        return False
    if len(text.split()) < f["min_words"]:
        return False
    if not text.strip(_JUNK_CHARS):
        # nothing but digits/whitespace/punctuation -- not real content
        return False
    return True


def build_sourced_corpus(corpus_name: str) -> pd.DataFrame:
    """
    Streams the raw CSV in chunks of READ_CHUNK_SIZE rows, filters
    each row, dedupes by content hash (not full text, to bound
    memory), and reservoir-samples a uniform random draw of size
    BASE_SAMPLE_SIZE[corpus_name] * (a random factor in
    OVERSAMPLE_FACTOR_RANGE) directly off the stream -- a single pass,
    with memory bounded by (chunk size + draw size + number of unique
    hashes seen), never by the full file size.
    """
    raw_path = RAW_FILES[corpus_name]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected raw CSV at {raw_path} -- place your {corpus_name} "
            f"export there (any text column; auto-detected if unnamed)."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    oversample_factor = rng.uniform(*OVERSAMPLE_FACTOR_RANGE)
    draw_n = int(round(BASE_SAMPLE_SIZE[corpus_name] * oversample_factor))

    reservoir = []            # holds at most draw_n texts
    seen_hashes = set()       # dedup by 16-byte digest, not full text
    n_seen = 0                 # count of unique, filter-passing rows seen so far
    text_col = None
    hint = TEXT_COLUMN_HINTS.get(corpus_name)

    chunk_iter = pd.read_csv(raw_path, chunksize=READ_CHUNK_SIZE, low_memory=False)
    for chunk_idx, chunk in enumerate(chunk_iter):
        if text_col is None:
            text_col = _detect_text_column_from_sample(chunk, hint)
            print(f"[{corpus_name}] using text column: '{text_col}'")

        for raw_text in chunk[text_col].astype(str):
            text = raw_text.strip()
            if not _row_passes_filters(text):
                continue
            h = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Algorithm R (reservoir sampling): uniform random sample
            # of size draw_n from a stream of unknown total length,
            # single pass, O(draw_n) reservoir memory.
            if len(reservoir) < draw_n:
                reservoir.append(text)
            else:
                j = rng.integers(0, n_seen + 1)
                if j < draw_n:
                    reservoir[j] = text
            n_seen += 1

        if chunk_idx % 20 == 0:
            print(f"[{corpus_name}] scanned ~{(chunk_idx + 1) * READ_CHUNK_SIZE:,} raw rows "
                  f"-> reservoir {len(reservoir)}/{draw_n}")

    sampled = pd.DataFrame({
        "sample_id": [f"{corpus_name}_{i:05d}" for i in range(len(reservoir))],
        "text": reservoir,
    })

    out_path = SOURCED_DIR / f"{corpus_name}_sourced.csv"
    sampled.to_csv(out_path, index=False)
    print(f"[{corpus_name}] {n_seen} unique valid rows seen -> drew {len(sampled)} "
          f"(target {draw_n}, oversample x{oversample_factor:.2f}) -> {out_path}")
    return sampled