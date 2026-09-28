"""
Classical (pre-LLM) baseline for privacy-risk classification: LOW / MEDIUM / HIGH.

BASELINE = Microsoft Presidio (standard open-source PII detector) on top of the spaCy
           `en_core_web_lg` NER model. No LLM, no training, no keyword lists, no tuning.

FIXED RULES (written from the LLM prompt's definitions BEFORE running on the evaluation set)
  HIGH   -> a critical identifier is detected: US_SSN, CREDIT_CARD, US_BANK_NUMBER, IBAN_CODE,
            US_PASSPORT, US_DRIVER_LICENSE, US_ITIN, CRYPTO, UK_NHS          (score >= 0.5)
  MEDIUM -> a contact identifier is detected: EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS, MAC_ADDRESS
            OR a PERSON appears together with a LOCATION / ORGANIZATION / NRP  (score >= 0.4)
  LOW    -> everything else.
  Do NOT change these rules or thresholds after seeing results on the evaluation set.

It reads ONLY the text column. Your true labels and your model's predictions are used only
to score the systems afterwards.

REPORTS (same 250 rows, same true labels) for three systems: always_LOW, Presidio_baseline, model
  accuracy, precision / recall / F1 (macro, weighted, micro), per-class precision / recall / F1,
  and the confusion matrix. Nothing else.

RUN
    pip install numpy pandas scikit-learn spacy presidio-analyzer
    python -m spacy download en_core_web_lg
    python presidio_baseline_eval.py --csv <your_csv>

OUTPUT (folder `baseline_results/`)
    report.txt              readable summary (share this)
    metrics_summary.csv     accuracy + macro / weighted / micro precision, recall, F1 per system
    per_class_metrics.csv   precision, recall, F1, support per class per system
    confusion_matrices.csv  counts per system
    rule_predictions.csv    per-sample: true, model, baseline prediction, what Presidio found
    results.json            everything above + settings, in one file
"""
import argparse
import json
import os
import sys
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# ---------------------------------------------------------------- fixed baseline rules
CRITICAL = {"US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE", "US_PASSPORT",
            "US_DRIVER_LICENSE", "US_ITIN", "CRYPTO", "UK_NHS"}
CONTACT = {"EMAIL_ADDRESS", "EMAIL", "PHONE_NUMBER", "IP_ADDRESS", "MAC_ADDRESS"}
LINKABLE = {"LOCATION", "ORGANIZATION", "NRP"}
MIN_SCORE_CRITICAL = 0.5
MIN_SCORE_OTHER = 0.4


def classify(results, text):
    crit, contact, persons, linkable = [], [], set(), set()
    for r in results:
        span = text[r.start:r.end].strip().lower()
        if r.entity_type in CRITICAL and r.score >= MIN_SCORE_CRITICAL:
            crit.append(r.entity_type)
        elif r.entity_type in CONTACT and r.score >= MIN_SCORE_OTHER:
            contact.append(r.entity_type)
        elif r.entity_type == "PERSON" and r.score >= MIN_SCORE_OTHER:
            persons.add(span)
        elif r.entity_type in LINKABLE and r.score >= MIN_SCORE_OTHER:
            linkable.add(span)

    if crit:
        label = "HIGH"
    elif contact or (persons and linkable):
        label = "MEDIUM"
    else:
        label = "LOW"
    counts = {"n_critical": len(crit), "n_contact": len(contact),
              "n_persons": len(persons), "n_linkable": len(linkable)}
    return label, counts, ";".join(sorted(set(crit)) + sorted(set(contact)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="survey_with_label3_predictions.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--true_col", default="label3")
    ap.add_argument("--model_pred_col", default="predicted_label")
    ap.add_argument("--spacy_model", default="en_core_web_lg")
    ap.add_argument("--out_dir", default="baseline_results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        cfg = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": args.spacy_model}]}
        analyzer = AnalyzerEngine(nlp_engine=NlpEngineProvider(nlp_configuration=cfg).create_engine(),
                                  supported_languages=["en"])
    except Exception as e:
        sys.exit(f"Could not start Presidio/spaCy ({type(e).__name__}: {e}).\n"
                 f"Run: pip install presidio-analyzer spacy && python -m spacy download {args.spacy_model}")

    names = ["LOW", "MEDIUM", "HIGH"]
    to_idx = {c: i for i, c in enumerate(names)}
    df = pd.read_csv(args.csv)
    if "prediction_status" in df.columns:
        df = df[df["prediction_status"] == "ok"]
    df = df.dropna(subset=[args.text_col, args.true_col, args.model_pred_col]).reset_index(drop=True)
    n = len(df)
    print(f"Loaded {n} rows. Running Presidio ({args.spacy_model}) - this can take a few minutes...")

    # ---- baseline predictions (reads ONLY the text) ----
    labels, counts, fired = [], [], []
    for i, text in enumerate(df[args.text_col].astype(str)):
        lab, c, f = classify(analyzer.analyze(text=text, language="en"), text)
        labels.append(lab); counts.append(c); fired.append(f)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}")

    y_true = df[args.true_col].str.strip().str.upper().map(to_idx).to_numpy().astype(int)
    systems = {
        "always_LOW": np.zeros(n, dtype=int),
        "Presidio_baseline": np.array([to_idx[l] for l in labels]),
        "model": df[args.model_pred_col].str.strip().str.upper().map(to_idx).to_numpy().astype(int),
    }
    ids = [0, 1, 2]

    # ---- metrics ----
    overall_rows, class_rows, cm_rows = [], [], []
    for s, yp in systems.items():
        row = {"system": s, "accuracy": accuracy_score(y_true, yp)}
        for avg in ["macro", "weighted", "micro"]:
            p, r, f, _ = precision_recall_fscore_support(y_true, yp, labels=ids, average=avg, zero_division=0)
            row.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
        overall_rows.append(row)

        p, r, f, sup = precision_recall_fscore_support(y_true, yp, labels=ids, average=None, zero_division=0)
        for i, nm in enumerate(names):
            class_rows.append({"system": s, "class": nm, "precision": p[i], "recall": r[i],
                               "f1": f[i], "support": int(sup[i])})
        cm = confusion_matrix(y_true, yp, labels=ids)
        for i, tn in enumerate(names):
            for j, pn in enumerate(names):
                cm_rows.append({"system": s, "true": tn, "pred": pn, "count": int(cm[i, j])})

    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(class_rows)
    cm_df = pd.DataFrame(cm_rows)

    # ---- save ----
    overall.to_csv(f"{args.out_dir}/metrics_summary.csv", index=False)
    per_class.to_csv(f"{args.out_dir}/per_class_metrics.csv", index=False)
    cm_df.to_csv(f"{args.out_dir}/confusion_matrices.csv", index=False)
    out = pd.DataFrame({"text_id": df["text_id"] if "text_id" in df.columns else np.arange(n),
                        "true": df[args.true_col].str.upper(), "model_pred": df[args.model_pred_col].str.upper(),
                        "baseline_pred": labels, "baseline_correct": systems["Presidio_baseline"] == y_true,
                        "model_correct": systems["model"] == y_true, "critical_or_contact_types": fired})
    out = pd.concat([out, pd.DataFrame(counts),
                     df[args.text_col].str.replace(r"\s+", " ", regex=True).str[:200].rename("text_preview")], axis=1)
    out.to_csv(f"{args.out_dir}/rule_predictions.csv", index=False)

    with open(f"{args.out_dir}/results.json", "w") as fh:
        json.dump({"run_info": {"timestamp": datetime.now().isoformat(timespec="seconds"), "csv": args.csv, "n": n,
                                "presidio_analyzer": version("presidio-analyzer"), "spacy": version("spacy"),
                                "spacy_model": args.spacy_model},
                   "rules": {"CRITICAL": sorted(CRITICAL), "CONTACT": sorted(CONTACT), "LINKABLE": sorted(LINKABLE),
                             "MIN_SCORE_CRITICAL": MIN_SCORE_CRITICAL, "MIN_SCORE_OTHER": MIN_SCORE_OTHER,
                             "HIGH": "any critical identifier",
                             "MEDIUM": "any contact identifier OR (PERSON and any of LOCATION/ORGANIZATION/NRP)",
                             "LOW": "otherwise"},
                   "overall_metrics": overall.to_dict("records"), "per_class_metrics": per_class.to_dict("records"),
                   "confusion_matrices_rows_true_cols_pred": {
                       s: confusion_matrix(y_true, yp, labels=ids).tolist() for s, yp in systems.items()},
                   "class_order": names}, fh, indent=2, default=float)

    # ---- readable report ----
    L = [f"Presidio baseline vs model | n = {n} | Presidio {version('presidio-analyzer')} + spaCy {args.spacy_model}",
         "Baseline rules are fixed and untuned.\n", "OVERALL METRICS"]
    L.append(overall.set_index("system").round(3).T.to_string() + "\n")
    L.append("Note: micro-averaged precision, recall and F1 all equal accuracy in single-label multiclass tasks.\n")
    L.append("PER-CLASS METRICS")
    for s in systems:
        L.append(f"[{s}]\n" + per_class[per_class.system == s].drop(columns="system")
                 .set_index("class").round(3).to_string() + "\n")
    L.append("CONFUSION MATRICES (rows = true, cols = predicted)")
    for s, yp in systems.items():
        L.append(f"[{s}]\n" + pd.DataFrame(confusion_matrix(y_true, yp, labels=ids),
                 index=[f"true_{x}" for x in names], columns=[f"pred_{x}" for x in names]).to_string() + "\n")
    report = "\n".join(L)
    open(f"{args.out_dir}/report.txt", "w", encoding="utf-8").write(report + "\n")
    print("\n" + report + f"\nSaved in {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
