import re
from typing import List, Set

from presidio_analyzer import (
    AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern,
    EntityRecognizer, RecognizerResult
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from rapidfuzz.distance import Levenshtein


# ---------- tokenization helpers (keep internal apostrophes & hyphens) ----------
NAME_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")  # e.g., O'Connor, Smith-Jones

def normalize_name_tokens(full_name: str) -> List[str]:
    """Extract tokens preserving internal ' ’ -, dropping 1-char initials."""
    return [t for t in NAME_TOKEN_RE.findall(full_name) if len(t) >= 2]

def strip_punct(s: str) -> str:
    """Remove apostrophes/hyphens for normalization."""
    return re.sub(r"['’-]", "", s)

def make_token_regex_allow_optional_punct(token: str) -> str:
    """
    Build a regex for a token where existing apostrophes/hyphens become optional.
    - "O'Connor" -> r"O['’-]?Connor"
    - "Smith-Jones" -> r"Smith['’-]?Jones"
    """
    parts = re.split(r"(['’-])", token)
    out = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p in ("'", "’", "-"):
            # make the separator optional
            out.append(r"['’-]?")
        else:
            out.append(re.escape(p))
        i += 1
    return "".join(out)

def build_deny_variants(tokens: List[str]) -> Set[str]:
    """
    Exact-match variants (case-insensitive), plus punctuation-dropped mirrors:
      - full name; first+last; each token alone
      - possessives on all variants
      - versions with apostrophes/hyphens removed
    """
    variants: Set[str] = set(tokens)

    if tokens:
        variants.add(" ".join(tokens))
    if len(tokens) >= 2:
        variants.add(f"{tokens[0]} {tokens[-1]}")

    # punctuation-dropped mirrors
    more = set()
    for v in list(variants):
        vv = strip_punct(v)
        if vv and vv != v:
            more.add(vv)
    variants |= more

    # possessives
    variants |= {v + "'s" for v in list(variants)}
    variants |= {v + "’s" for v in list(variants)}
    return variants


# ---------- fuzzy recognizer (len >= 3); normalize both sides ----------
class FuzzySubjectNameRecognizer(EntityRecognizer):
    def __init__(self, subject_tokens: List[str]):
        super().__init__(supported_entities=["SUBJECT_NAME"], supported_language="en")
        # store both original and punctuation-stripped forms
        self.targets_raw = [t for t in subject_tokens if len(t) >= 3]
        self.targets_norm = [strip_punct(t).lower() for t in self.targets_raw]

    def load(self):  # no-op
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        if "SUBJECT_NAME" not in entities or not self.targets_raw:
            return []

        results = []
        # token with optional possessive; preserve internal punctuation
        for m in re.finditer(r"\b([A-Za-z]+(?:['’-][A-Za-z]+)*)(?:['’]s)?\b", text):
            core = m.group(1)
            if len(core) < 3:
                continue
            lc_raw = core.lower()
            lc_norm = strip_punct(lc_raw)

            # Compare both raw and normalized (punct dropped) forms
            matched = False
            for tgt_raw, tgt_norm in zip(self.targets_raw, self.targets_norm):
                if (Levenshtein.distance(lc_raw, tgt_raw.lower()) <= 1 or
                    Levenshtein.distance(lc_norm, tgt_norm) <= 1):
                    score = 1.0 if (lc_raw == tgt_raw.lower() or lc_norm == tgt_norm) else 0.9
                    results.append(RecognizerResult("SUBJECT_NAME", m.start(), m.end(), score))
                    matched = True
                    break
            if matched:
                continue
        return results


# ---------- regex recognizer to eat title + middle initials + last ----------
# def build_title_name_pattern(tokens: List[str]) -> str:
#     """
#     Build a regex that matches:
#       [optional 'Dr' prefix] + First + optional middle initial(s) + Last + optional possessive
#     - allows apostrophes/hyphens in tokens, and treats them as optional where present
#     - middle initials: one or two single-letter tokens with optional dot
#     """
#     if len(tokens) < 2:
#         return ""  # need at least first & last

#     first_pat = make_token_regex_allow_optional_punct(tokens[0])
#     last_pat  = make_token_regex_allow_optional_punct(tokens[-1])

#     # up to two middle initials like "P." / "Q" with flexible spacing
#     mid_initial = r"(?:\s+[A-Za-z]\.?\s*)"
#     mid_block = fr"{mid_initial}{{0,2}}"   # optional up to 2

#     # optional "Dr" prefix (with or without dot), consume following spaces
#     prefix = r"(?:\b(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.?\s*)?"

#     # optional possessive after the last name
#     poss = r"(?:['’]s)?"

#     # assemble
#     pat = fr"{prefix}\b{first_pat}{mid_block}\s+{last_pat}{poss}\b"
#     return pat

def build_title_name_pattern(tokens: List[str]) -> str:
    """
    Build a regex that matches:
      [optional title] + First + optional middle initial(s) + Last(+compound) + optional possessive
    Also supports: [optional title] + Last(+compound) (+possessive)
    - allows apostrophes/hyphens in tokens, and treats them as optional where present
    - middle initials: one or two single-letter tokens with optional dot
    - spacing between parts can be zero or more (handles 'DrJohnPOConnor')
    """
    if len(tokens) < 2:
        return ""  # need at least first & last from the subject name

    first_pat = make_token_regex_allow_optional_punct(tokens[0])
    last_pat  = make_token_regex_allow_optional_punct(tokens[-1])

    # up to two middle initials like "P." / "Q" with flexible/optional spacing
    mid_initial = r"(?:\s*[A-Za-z]\.?\s*)"
    mid_block = fr"{mid_initial}{{0,2}}"   # optional up to 2

    # optional title prefix (with/without dot), allow zero-or-more whitespace after
    prefix = r"(?:\b(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.?\s*)?"

    # allow compound surname parts after the last name (e.g., "-Smith", "’Reilly")
    compound_suffix = r"(?:['’-][A-Za-z]+)*"

    # optional possessive after the (possibly compound) last name
    poss = r"(?:['’]s)?"

    # Full pattern: [prefix] First [mid initials] Last(+compound) [poss]
    pat_full = fr"{prefix}{first_pat}{mid_block}\s*{last_pat}{compound_suffix}{poss}\b"

    # Also match: [prefix] Last(+compound) [poss]  (covers 'Prof O'Connor', 'Dr OConnor's')
    pat_lastonly = fr"{prefix}{last_pat}{compound_suffix}{poss}\b"

    # Combine
    return fr"(?:{pat_full}|{pat_lastonly})"


class TitleAndInitialsRecognizer(PatternRecognizer):
    """
    Pattern recognizer for 'Dr. First M. Last' (including variants):
    - Includes 'Dr' in the matched span so it gets redacted too
    - Consumes 0-2 middle initials (single letters, with/without '.')
    - Allows dropped/optional apostrophes/hyphens inside tokens
    """
    def __init__(self, tokens: List[str]):
        pat = build_title_name_pattern(tokens)
        patterns = [Pattern(name="title_first_midinit_last", regex=pat, score=0.9)] if pat else []
        super().__init__(supported_entity="SUBJECT_NAME", name="subject_title_initials", patterns=patterns)


# ---------- build & run ----------
def build_presidio():
    nlp_conf = {"nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry), AnonymizerEngine(), registry

def add_subject_name_detectors(registry: RecognizerRegistry, subject_full_name: str):
    tokens = normalize_name_tokens(subject_full_name)
    if not tokens:
        return

    # 1) exact matches + punctuation-dropped mirrors (incl. possessives)
    deny_variants = list(build_deny_variants(tokens))
    registry.add_recognizer(PatternRecognizer(
        supported_entity="SUBJECT_NAME",
        deny_list=deny_variants,
        name="subject_name_denylist",
    ))

    # 2) title + initials + last (regex) to eat "Dr." and middle initials in span
    registry.add_recognizer(TitleAndInitialsRecognizer(tokens))

    # 3) fuzzy token matcher (len >= 3), Levenshtein <= 1 on raw and punctuation-dropped
    registry.add_recognizer(FuzzySubjectNameRecognizer(tokens))


def redact_subject_name(text: str, subject_full_name: str, replacement: str = "[REDACTED-NAME]") -> str:
    analyzer, anonymizer, registry = build_presidio()
    add_subject_name_detectors(registry, subject_full_name)
    results = analyzer.analyze(text=text, entities=["SUBJECT_NAME"], language="en")
    operators = {"SUBJECT_NAME": OperatorConfig("replace", {"new_value": replacement})}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


# ---------- example ----------
if __name__ == "__main__":
    subject = "John P. O'Connor"
    sample = (
        "Smith-Jones examined the patient Dr. John P. O'Connor. O’Connor's note mentions follow-up. "
        "The lab mislabeled it as OConor yesterday. Also saw John alone. Then I Drove home"
    )
    print(redact_subject_name(sample, subject))
    # Expected redactions:
    # - "Dr. John P. O'Connor"  (includes Dr. and P.)
    # - "O’Connor's"
    # - "OConor"  (dropped apostrophe, edit distance 1)
    # - "John"
