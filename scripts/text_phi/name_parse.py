"""Parse a free-text subject-name cell into a `PersonalName`.

Supports an optional `name_format` hint plus a permissive default. Order among
first / middle / last is only load-bearing for the subject-name title-and-
initials regex ("Dr. J. L. Smith"); the deny-list and fuzzy layers treat all
tokens equally, so a wrong parse degrades gracefully.
"""

from __future__ import annotations

import re
from typing import Literal

from clean_eeg.anonymize import PersonalName


NameFormat = Literal["first last", "first middle last", "last, first", "last, first middle"]

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*\.?")


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def _strip_trailing_dot(t: str) -> str:
    return t[:-1] if t.endswith(".") else t


def parse_name(cell: str, name_format: str | None = None) -> PersonalName:
    """Parse a name cell into a PersonalName.

    Parameters
    ----------
    cell: raw text from the name column
    name_format: optional layout hint; one of
        "first last", "first middle last", "last, first", "last, first middle".
        If None, layout is inferred: cells containing a comma are treated as
        "last, first middle"; otherwise as "first [middle...] last".

    Single-token cells are returned as last-name-only PersonalNames. The
    subject redactor's `Title? Last` regex branch handles these.
    """
    if cell is None or not cell.strip():
        raise ValueError("Empty name cell")

    raw_tokens = _tokens(cell)
    tokens = [_strip_trailing_dot(t) for t in raw_tokens if t.strip(".")]
    if not tokens:
        raise ValueError(f"No name tokens in cell: {cell!r}")

    if name_format is not None:
        return _parse_with_format(cell, tokens, name_format)
    return _parse_permissive(cell, tokens)


def _parse_permissive(cell: str, tokens: list[str]) -> PersonalName:
    if len(tokens) == 1:
        return PersonalName(first_name="", middle_names=[], last_name=tokens[0])

    if "," in cell:
        # "Last, First [Middle...]": last comes before the first comma; the
        # rest is first + middles.
        last_side, _, first_side = cell.partition(",")
        last_tokens = [_strip_trailing_dot(t) for t in _tokens(last_side)]
        first_tokens = [_strip_trailing_dot(t) for t in _tokens(first_side)]
        if not last_tokens or not first_tokens:
            # Comma but empty on one side — fall back to whitespace order.
            return PersonalName(
                first_name=tokens[0],
                middle_names=tokens[1:-1],
                last_name=tokens[-1],
            )
        last = last_tokens[-1]  # in case of "Smith Jr., John"
        first = first_tokens[0]
        middles = first_tokens[1:] + last_tokens[:-1]
        return PersonalName(first_name=first, middle_names=middles, last_name=last)

    return PersonalName(
        first_name=tokens[0],
        middle_names=tokens[1:-1],
        last_name=tokens[-1],
    )


def _parse_with_format(cell: str, tokens: list[str], name_format: str) -> PersonalName:
    fmt = name_format.strip().lower()
    if fmt in ("first last", "firstlast"):
        if len(tokens) < 2:
            return PersonalName(first_name="", middle_names=[], last_name=tokens[-1])
        return PersonalName(first_name=tokens[0], middle_names=[], last_name=tokens[-1])
    if fmt in ("first middle last", "firstmiddlelast"):
        if len(tokens) < 2:
            return PersonalName(first_name="", middle_names=[], last_name=tokens[-1])
        return PersonalName(
            first_name=tokens[0],
            middle_names=tokens[1:-1],
            last_name=tokens[-1],
        )
    if fmt in ("last, first", "last,first"):
        return _parse_last_comma_first(cell, tokens, with_middles=False)
    if fmt in ("last, first middle", "last,first middle", "last,firstmiddle"):
        return _parse_last_comma_first(cell, tokens, with_middles=True)
    raise ValueError(
        f"Unknown name_format {name_format!r}. Use one of: "
        "'first last', 'first middle last', 'last, first', 'last, first middle'."
    )


def _parse_last_comma_first(cell: str, tokens: list[str], with_middles: bool) -> PersonalName:
    if "," in cell:
        last_side, _, first_side = cell.partition(",")
        last_tokens = [_strip_trailing_dot(t) for t in _tokens(last_side)]
        first_tokens = [_strip_trailing_dot(t) for t in _tokens(first_side)]
        if not last_tokens:
            last_tokens = [tokens[0]]
            first_tokens = tokens[1:]
        elif not first_tokens:
            first_tokens = [tokens[-1]]
        last = last_tokens[-1]
        first = first_tokens[0]
        middles: list[str] = []
        if with_middles:
            middles = first_tokens[1:] + last_tokens[:-1]
        return PersonalName(first_name=first, middle_names=middles, last_name=last)
    # No comma at all — fall back to "first middle last" layout.
    if len(tokens) == 1:
        return PersonalName(first_name="", middle_names=[], last_name=tokens[0])
    return PersonalName(
        first_name=tokens[0],
        middle_names=tokens[1:-1] if with_middles else [],
        last_name=tokens[-1],
    )
