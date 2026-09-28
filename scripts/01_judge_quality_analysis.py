#!/usr/bin/env python3
"""
01_judge_quality_analysis.py

Per-corpus judge quality analysis for 3-way LLM-judge label CSVs
(labels expected to be one of: LOW / MEDIUM / HIGH, plus an optional UNKNOWN).

For a single corpus, this script:
  1. Loads the three judge CSVs and merges them on sample_id (verifies text
     is identical across files).
  2. Reports label distributions and parse-status distributions per judge.
  3. Computes pairwise raw agreement, Cohen's kappa (unweighted + quadratic
     weighted), and Pearson/Spearman correlation (ordinal LOW=1/MED=2/HIGH=3).
  4. Prints full pairwise crosstabs.
  5. Runs a generic, corpus-agnostic "signal detector" (regex + bracket-tag
     PII entities) to flag rows that plausibly contain critical sensitive
     info (SSN/credit-card/credential/account/medical/legal) vs. rows with
     no detectable personal/sensitive signal at all, then reports each
     judge's HIGH-rate / elevated-rate on both groups as a sanity check
     against over/under-labeling.
  6. Saves one merged CSV per corpus with all raw labels + boolean signal
     columns, for reuse by the other two scripts.

USAGE
-----
    python 01_judge_quality_analysis.py \
        --judge-a path/to/xyz_judge_a.csv \
        --judge-b path/to/xyz_judge_b.csv \
        --judge-c path/to/xyz_judge_c.csv \
        --corpus-name xyz \
        --outdir ./out

If --corpus-name is omitted, it's inferred from the judge-a filename
(text before "_judge_a").

Requires: pandas, numpy, scipy, scikit-learn
    pip install pandas numpy scipy scikit-learn
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]
ORDINAL_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

# ---------------------------------------------------------------------------
# Generic, corpus-agnostic "does this text contain sensitive personal info?"
# heuristic. Two independent detectors are combined:
#   (a) bracket-tag PII entities, e.g. "[123-45-6789]ssn"  (nemotron-style)
#   (b) plain-text regex signals (emails, phones, credentials, etc.)
# This is NOT a ground truth -- it's a sanity-check signal to see whether a
# judge's HIGH/MEDIUM label correlates with plausible sensitive content.
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"\]([a-zA-Z_0-9]+)")

CRITICAL_TAGS = {
    "ssn", "national_id", "tax_id", "credit_debit_card", "cvv",
    "bank_routing_number", "account_number", "swift_bic",
    "medical_record_number", "health_plan_beneficiary_number",
    "biometric_identifier", "password", "pin", "api_key",
    "blood_type", "sexuality", "religious_belief", "political_view",
    "race_ethnicity", "passport_number",
}
IDENT_TAGS = {
    "first_name", "last_name", "email", "phone_number", "street_address",
    "customer_id", "employee_id", "user_name", "ipv4", "ipv6",
    "device_identifier", "mac_address", "coordinate", "postcode",
    "license_plate", "vehicle_identifier", "http_cookie", "unique_id",
    "certificate_license_number", "education_level", "employment_status",
    "gender", "age", "language", "occupation", "county", "city", "state",
    "date_of_birth", "fax_number",
}

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CRED_RE = re.compile(
    r"\b(password|pwd|passcode|pin\b|login ?id|user ?id|ssn|social security)\b",
    re.I,
)
ACCOUNT_RE = re.compile(
    r"\b(account\s*(number|#|no\.?)|acct\s*#|routing number)\b", re.I
)
MEDICAL_RE = re.compile(
    r"\b(diagnos\w*|medical|cancer|illness|surger\w*|prescription|therapy|disability)\b",
    re.I,
)
LEGAL_SALARY_RE = re.compile(
    r"\b(lawsuit|subpoena|litigation|deposition|indict\w*|allegation|settlement|"
    r"salary|bonus|compensation|severance|wages)\b",
    re.I,
)


def detect_signals(text: str) -> dict:
    """Return a dict of boolean signal flags for one document."""
    text = text if isinstance(text, str) else ""
    tags = set(TAG_RE.findall(text))
    has_critical_tag = bool(tags & CRITICAL_TAGS)
    has_ident_tag = bool(tags & IDENT_TAGS)

    has_email = bool(EMAIL_RE.search(text))
    has_phone = bool(PHONE_RE.search(text))
    has_ssn = bool(SSN_RE.search(text))
    has_cred = bool(CRED_RE.search(text))
    has_account = bool(ACCOUNT_RE.search(text))
    has_medical = bool(MEDICAL_RE.search(text))
    has_legal_salary = bool(LEGAL_SALARY_RE.search(text))

    critical_signal = has_critical_tag or has_ssn or has_cred or has_account or has_medical
    any_signal = (
        critical_signal
        or has_ident_tag
        or has_email
        or has_phone
        or has_legal_salary
    )
    return {
        "has_email": has_email,
        "has_phone": has_phone,
        "has_ssn": has_ssn,
        "has_credential_kw": has_cred,
        "has_account_kw": has_account,
        "has_medical_kw": has_medical,
        "has_legal_salary_kw": has_legal_salary,
        "has_critical_tag": has_critical_tag,
        "has_ident_tag": has_ident_tag,
        "critical_signal": critical_signal,
        "any_signal": any_signal,
    }


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
    keep = ["sample_id", "text", f"{generic_name}_label", f"{generic_name}_parse_status"]
    return df[keep]


def kappa_pair(m: pd.DataFrame, col_x: str, col_y: str):
    if cohen_kappa_score is None:
        return None, None
    sub = m[m[col_x].isin(LABEL_ORDER) & m[col_y].isin(LABEL_ORDER)]
    k = cohen_kappa_score(sub[col_x], sub[col_y])
    wk = cohen_kappa_score(sub[col_x], sub[col_y], weights="quadratic")
    return k, wk


def corr_pair(m: pd.DataFrame, col_x: str, col_y: str):
    if pearsonr is None or spearmanr is None:
        return None, None
    sub = m.dropna(subset=[col_x, col_y])
    sp = spearmanr(sub[col_x], sub[col_y]).correlation
    pe = pearsonr(sub[col_x], sub[col_y])[0]
    return pe, sp


def main():
    ap = argparse.ArgumentParser(description="Per-corpus judge quality analysis")
    ap.add_argument("--judge-a", required=True, help="CSV for judge A")
    ap.add_argument("--judge-b", required=True, help="CSV for judge B")
    ap.add_argument("--judge-c", required=True, help="CSV for judge C")
    ap.add_argument("--corpus-name", default=None, help="Name used in output filenames")
    ap.add_argument("--outdir", default=".", help="Directory to write outputs to")
    args = ap.parse_args()

    corpus_name = args.corpus_name or os.path.basename(args.judge_a).split("_judge_a")[0]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*70}\nCORPUS: {corpus_name}\n{'='*70}")

    a = load_judge(args.judge_a, "judge_a")
    b = load_judge(args.judge_b, "judge_b")
    c = load_judge(args.judge_c, "judge_c")

    # ---- sanity checks ----
    assert set(a.sample_id) == set(b.sample_id) == set(c.sample_id), "sample_id sets differ!"
    m = a.merge(b[["sample_id", "judge_b_label", "judge_b_parse_status"]], on="sample_id")
    m = m.merge(c[["sample_id", "judge_c_label", "judge_c_parse_status"]], on="sample_id")
    assert len(m) == len(a), "merge changed row count -- check for duplicate sample_ids"

    # ---- 1. label / parse-status distributions ----
    print("\n--- Label distributions ---")
    for j in ["judge_a", "judge_b", "judge_c"]:
        print(f"\n{j}:")
        print(m[f"{j}_label"].value_counts(dropna=False))

    print("\n--- Parse status ---")
    for j in ["judge_a", "judge_b", "judge_c"]:
        vc = m[f"{j}_parse_status"].value_counts(dropna=False)
        print(f"{j}: {dict(vc)}")

    # ---- 2. agreement / kappa / correlation ----
    for j in ["judge_a", "judge_b", "judge_c"]:
        m[f"{j}_num"] = m[f"{j}_label"].map(ORDINAL_MAP)

    pairs = [("judge_a", "judge_b"), ("judge_a", "judge_c"), ("judge_b", "judge_c")]
    print("\n--- Pairwise agreement / kappa / correlation ---")
    for x, y in pairs:
        raw_agree = (m[f"{x}_label"] == m[f"{y}_label"]).mean()
        k, wk = kappa_pair(m, f"{x}_label", f"{y}_label")
        pe, sp = corr_pair(m, f"{x}_num", f"{y}_num")
        print(
            f"{x} vs {y}: raw_agree={raw_agree:.3f}  kappa={k:.3f}  "
            f"weighted_kappa={wk:.3f}  pearson_r={pe:.3f}  spearman_rho={sp:.3f}"
        )

    all3 = (m.judge_a_label == m.judge_b_label) & (m.judge_b_label == m.judge_c_label)
    print(f"\nAll-3-agree rate: {all3.mean():.3f}")

    # ---- 3. crosstabs ----
    print("\n--- Crosstabs ---")
    for x, y in pairs:
        print(f"\n{x} (rows) vs {y} (cols):")
        print(pd.crosstab(m[f"{x}_label"], m[f"{y}_label"]))

    # ---- 4. calibration / signal-detector sanity check ----
    print("\n--- Calibration check (generic sensitive-content signal) ---")
    sig_df = m.text.apply(detect_signals).apply(pd.Series)
    m = pd.concat([m, sig_df], axis=1)

    n_critical = int(m.critical_signal.sum())
    n_plain = int((~m.any_signal).sum())
    print(f"Rows flagged with a critical sensitive signal: {n_critical}")
    print(f"Rows with NO detectable personal/sensitive signal at all: {n_plain}")

    for j in ["judge_a", "judge_b", "judge_c"]:
        col = f"{j}_label"
        crit = m[m.critical_signal]
        plain = m[~m.any_signal]
        high_crit = (crit[col] == "HIGH").mean() if len(crit) else float("nan")
        elevated_crit = crit[col].isin(["MEDIUM", "HIGH"]).mean() if len(crit) else float("nan")
        high_plain = (plain[col] == "HIGH").mean() if len(plain) else float("nan")
        elevated_plain = plain[col].isin(["MEDIUM", "HIGH"]).mean() if len(plain) else float("nan")
        print(
            f"{j}: on CRITICAL-signal rows -> HIGH={high_crit:.1%} MED+HIGH={elevated_crit:.1%} | "
            f"on PLAIN rows -> HIGH={high_plain:.1%} MED+HIGH={elevated_plain:.1%}"
        )

    # ---- 5. save merged output for reuse by scripts 2 & 3 ----
    out_path = os.path.join(args.outdir, f"{corpus_name}_merged.csv")
    m.to_csv(out_path, index=False)
    print(f"\nSaved merged file -> {out_path}")


if __name__ == "__main__":
    main()
