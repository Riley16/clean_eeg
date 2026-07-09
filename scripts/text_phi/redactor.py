"""TextRedactor: unify subject-name + generic-PHI spans for arbitrary text.

Reuses `clean_eeg.anonymize.SubjectNameRedactor` unchanged. For span-level
output (needed for the audit report) we call its `.analyzer.analyze()`
directly rather than `.redact()`, which returns a redacted string only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from clean_eeg.anonymize import PersonalName, SubjectNameRedactor

from .detectors import build_generic_analyzer, filter_person_hits_by_whitelist


Mode = Literal["subject", "generic", "both"]
ReplacementStyle = Literal["literal", "labeled"]


@dataclass(frozen=True)
class RedactionSpan:
    start: int
    end: int
    entity_type: str
    score: float
    recognizer: str
    matched_text: str


@dataclass
class RedactionResult:
    text: str
    spans: list[RedactionSpan] = field(default_factory=list)


def _result_to_span(r: RecognizerResult, source_text: str) -> RedactionSpan:
    return RedactionSpan(
        start=r.start,
        end=r.end,
        entity_type=r.entity_type,
        score=r.score,
        recognizer=getattr(r, "recognition_metadata", {}).get("recognizer_name", "")
                    if getattr(r, "recognition_metadata", None) else "",
        matched_text=source_text[r.start:r.end],
    )


def _merge_spans(spans: list[RedactionSpan]) -> list[RedactionSpan]:
    """Resolve overlapping spans. Higher score wins; ties favor SUBJECT_NAME."""
    if not spans:
        return []

    def sort_key(s: RedactionSpan) -> tuple:
        subject_bonus = 0 if s.entity_type == "SUBJECT_NAME" else 1
        return (s.start, -s.score, subject_bonus, s.end)

    ordered = sorted(spans, key=sort_key)
    kept: list[RedactionSpan] = []
    for s in ordered:
        if not kept:
            kept.append(s)
            continue
        prev = kept[-1]
        if s.start >= prev.end:
            kept.append(s)
            continue
        # Overlap: keep the higher-scoring one; on tie, prefer SUBJECT_NAME,
        # then the wider span.
        challenger_wins = (
            s.score > prev.score
            or (s.score == prev.score
                and s.entity_type == "SUBJECT_NAME"
                and prev.entity_type != "SUBJECT_NAME")
            or (s.score == prev.score
                and s.entity_type == prev.entity_type
                and (s.end - s.start) > (prev.end - prev.start))
        )
        if challenger_wins:
            kept[-1] = s
    return kept


def _apply_replacements(
    text: str,
    spans: list[RedactionSpan],
    style: ReplacementStyle,
    literal_replacement: str,
) -> str:
    """Apply replacements right-to-left so earlier offsets stay valid."""
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        repl = literal_replacement if style == "literal" else f"[{s.entity_type}]"
        text = text[:s.start] + repl + text[s.end:]
    return text


class TextRedactor:
    """Composite redactor across subject-name and generic PHI layers.

    Parameters
    ----------
    mode: which layers to run.
    subject_names: pre-configured subjects (used when mode is subject/both and
        no per-call override is supplied to `redact_with_subject`).
    replacement_style: "literal" (default) replaces every span with
        `literal_replacement`; "labeled" replaces with `"[<ENTITY>]"`.
    enable_zip / enable_age / mrn_regex: forwarded to `build_generic_analyzer`.
    entities: optional whitelist of entity types to keep from the generic
        analyzer; None means all.
    """

    def __init__(
        self,
        mode: Mode,
        subject_names: Iterable[PersonalName] | None = None,
        replacement_style: ReplacementStyle = "literal",
        literal_replacement: str = "X",
        enable_zip: bool = True,
        enable_age: bool = True,
        mrn_regex: str | None = None,
        entities: list[str] | None = None,
    ):
        if mode not in ("subject", "generic", "both"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode: Mode = mode
        self.replacement_style: ReplacementStyle = replacement_style
        self.literal_replacement = literal_replacement
        self.entities = entities

        self._subject_cache: dict[str, SubjectNameRedactor] = {}
        for pn in subject_names or []:
            self._get_subject_redactor(pn)

        if mode in ("generic", "both"):
            self._generic: AnalyzerEngine | None = build_generic_analyzer(
                enable_zip=enable_zip, enable_age=enable_age, mrn_regex=mrn_regex
            )
        else:
            self._generic = None

    # ---------- subject-redactor cache ----------

    def _get_subject_redactor(self, pn: PersonalName) -> SubjectNameRedactor:
        key = pn.get_full_name().lower()
        red = self._subject_cache.get(key)
        if red is None:
            red = SubjectNameRedactor(pn, replacement=self.literal_replacement)
            self._subject_cache[key] = red
        return red

    # ---------- redaction ----------

    def redact(self, text: str) -> RedactionResult:
        """Redact using pre-configured subject names (if any) + generic layer."""
        if not text:
            return RedactionResult(text=text, spans=[])
        subjects = list(self._subject_cache.values()) if self.mode != "generic" else []
        return self._redact_with(text, subjects)

    def redact_with_subject(
        self, text: str, subject_name: PersonalName
    ) -> RedactionResult:
        """Redact using a per-call subject (typical CSV per-row usage)."""
        if not text:
            return RedactionResult(text=text, spans=[])
        if self.mode == "generic":
            return self._redact_with(text, [])
        return self._redact_with(text, [self._get_subject_redactor(subject_name)])

    def _redact_with(
        self, text: str, subject_redactors: list[SubjectNameRedactor]
    ) -> RedactionResult:
        spans: list[RedactionSpan] = []

        # Subject-name layer.
        for red in subject_redactors:
            results = red.analyzer.analyze(
                text=text, entities=["SUBJECT_NAME"], language="en"
            )
            spans.extend(_result_to_span(r, text) for r in results)

        # Generic layer.
        if self._generic is not None:
            gen_results = self._generic.analyze(
                text=text, entities=self.entities, language="en"
            )
            gen_results = filter_person_hits_by_whitelist(text, gen_results)
            spans.extend(_result_to_span(r, text) for r in gen_results)

        merged = _merge_spans(spans)
        redacted = _apply_replacements(
            text, merged, self.replacement_style, self.literal_replacement
        )
        return RedactionResult(text=redacted, spans=merged)
