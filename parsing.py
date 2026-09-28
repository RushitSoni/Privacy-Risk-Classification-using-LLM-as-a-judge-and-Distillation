"""
Shared output parser for judge responses. All three judges are
prompted for the exact same label-only format, so one parser
covers all of them (mirrors the fallback-pattern design from the
original stage-1 notebook).

On the Transformers backend, judge_a (gpt-oss) normally returns only
its FINAL channel here, but if generation is truncated mid-reasoning
the container falls back to handing us the analysis text instead. So
the loosest pattern below scans for the LAST bare label mention
rather than the first -- in a reasoning trace the conclusion comes at
the end, and an early "...this is not LOW risk..." should not win.
"""
import re

# from config import VALID_LABELS
from config_small import VALID_LABELS

LABEL_PATTERNS = [
    re.compile(r"Label\s*:\s*(LOW|MEDIUM|HIGH)", re.IGNORECASE),
    re.compile(r"^(LOW|MEDIUM|HIGH)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b(LOW|MEDIUM|HIGH)\b", re.IGNORECASE),
]


def parse_label(raw_output: str):
    """
    Returns (label, status) where status is 'ok' or 'fallback_unknown'.
    Falls through progressively looser patterns rather than
    discarding a response outright.
    """
    text = (raw_output or "").strip()
    for i, pattern in enumerate(LABEL_PATTERNS):
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        # Strict patterns: take the first hit. Loosest bare-word
        # pattern (the last one): take the final hit.
        m = matches[-1] if i == len(LABEL_PATTERNS) - 1 else matches[0]
        label = m.group(1).upper()
        if label in VALID_LABELS:
            return label, "ok"
    return "UNKNOWN", "fallback_unknown"
