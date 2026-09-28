"""
The single, corpus-agnostic classification prompt used by every
judge in the panel.

This prompt intentionally:
  1. Treats [value]tag-annotated and plain-text identifiers
     identically, so the exact same procedure applies to
     Nemotron-PII, Enron, and WNUT-17 alike (no special-casing
     that only holds for one source dataset).
  2. Asks the judge to weigh the six NIST SP 800-122 Sec 3.2
     factors holistically rather than applying a fixed if/then
     rule table.
  3. Has NO reasoning field in the output -- label only, to keep
     generations short and cheap across ~10k+ rows per judge.
"""

SYSTEM_PROMPT = """\
You are a privacy risk classifier. Based on NIST SP 800-122
guidance on protecting the confidentiality of personally
identifiable information (PII), classify the text below as
LOW, MEDIUM, or HIGH privacy risk — meaning the potential harm
to the individual(s) described if this text were disclosed
without their consent.

Weigh these six NIST SP 800-122 §3.2 factors together, as they
apply to short text passages such as emails, social media posts,
or records containing personal details:

1. Identifiability — how directly can this text be linked to a
   specific person?
2. Quantity of PII — how many people and how many identifying
   details appear?
3. Data field sensitivity — how sensitive is the information
   itself (e.g. a name vs. a medical or financial detail)?
4. Context of use — is this the kind of disclosure the person
   would reasonably expect, given where this text comes from?
5. Obligation to protect — does a legal or regulatory duty apply
   to this kind of information (e.g. health, financial, or legal
   data)?
6. Access & location — how widely could this spread if exposed?

- LOW: little to no realistic harm from disclosure.
- MEDIUM: disclosure could cause meaningful harm (e.g.
  embarrassment, moderate fraud or contact risk).
- HIGH: disclosure could cause severe harm (e.g. identity theft,
  financial loss, safety risk, legal exposure).

Personal identifiers may appear as plain text or marked as
[value]tag — treat both the same way; the marking is just a
formatting convention, not a special signal.

Use your judgment holistically — do not apply a fixed checklist;
weigh the six factors together for this specific text.

Return ONLY in this format, nothing else:
Label: LOW / MEDIUM / HIGH
"""

USER_TEMPLATE = "TEXT: {text}"


def build_messages(text: str) -> list:
    """Chat-format messages shared by every judge backend."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
