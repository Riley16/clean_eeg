"""Tests for the per-site annotation-boilerplate whitelist that
controls what makes it into the end-of-run 'Human review needed' block.
"""

from __future__ import annotations

import json

import pytest

from clean_eeg.annotation_boilerplate import (
    BoilerplateWhitelist,
    BoilerplateWhitelistError,
    load_whitelist,
)


def test_missing_path_returns_empty_whitelist(tmp_path):
    wl = load_whitelist(tmp_path / "does_not_exist.json")
    assert wl.shared == []
    assert wl.per_site == {}
    # Empty whitelist never matches — safe fallback.
    assert not wl.matches("anything", site_code="A")


def test_none_path_returns_empty_whitelist():
    wl = load_whitelist(None)
    assert not wl.matches("anything")


def test_loads_shared_and_per_site(tmp_path):
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [r"^\s*NIGHT\s+CHECK\s*$"],
        "per_site": {"A": [r"^\s*TECH:.*$"], "S": [r"^\s*STIM ON\s*$"]},
    }))
    wl = load_whitelist(path)
    assert wl.matches("NIGHT CHECK", site_code="A")
    assert wl.matches("  NIGHT CHECK  ", site_code="J")  # shared applies everywhere
    assert wl.matches("TECH: initials MK", site_code="A")
    assert not wl.matches("TECH: initials MK", site_code="S")  # A-specific
    assert wl.matches("STIM ON", site_code="S")


def test_unknown_site_falls_through_to_shared(tmp_path):
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [r"^SYSTEM$"],
        "per_site": {"A": [r"^A-ONLY$"]},
    }))
    wl = load_whitelist(path)
    # Site "Z" not in the map — shared still applies, per-site does not.
    assert wl.matches("SYSTEM", site_code="Z")
    assert not wl.matches("A-ONLY", site_code="Z")


def test_none_site_uses_shared_only(tmp_path):
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [r"^SHARED$"],
        "per_site": {"A": [r"^A-ONLY$"]},
    }))
    wl = load_whitelist(path)
    assert wl.matches("SHARED", site_code=None)
    assert not wl.matches("A-ONLY", site_code=None)


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not: valid json")
    with pytest.raises(BoilerplateWhitelistError, match="not valid JSON"):
        load_whitelist(path)


def test_wrong_top_level_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(BoilerplateWhitelistError, match="top-level must be"):
        load_whitelist(path)


def test_invalid_regex_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"shared": ["["], "per_site": {}}))
    with pytest.raises(BoilerplateWhitelistError, match="invalid regex"):
        load_whitelist(path)


def test_default_shipped_whitelist_loads(tmp_path):
    """The shipped whitelist at data/annotation_boilerplate_whitelist.json
    must load cleanly and match its documented per-site phrases."""
    from clean_eeg.paths import ANNOTATION_BOILERPLATE_WHITELIST_PATH
    wl = load_whitelist(ANNOTATION_BOILERPLATE_WHITELIST_PATH)
    assert isinstance(wl, BoilerplateWhitelist)

    # CUDA (site 'A') boilerplate phrases from the shipped file must
    # fullmatch what R1652A / R1659A's annotation audits actually
    # flagged.
    for phrase in ("PAT REF EEG", "PAT BIPOLAR EEG", "PAT BP_II EEG",
                    "PAT REF_II EEG", "PAT ALL REF EEG",
                    "CAL IN", "E/C LAYING ON L. SID",
                    "E/C LAYING ON R. SID"):
        assert wl.matches(phrase, site_code="A"), (
            f"shipped CUDA whitelist should match {phrase!r}"
        )

    # Stim Start / Stim Stop patterns accept an optional single-word
    # suffix with either dash or space separator.
    for phrase in ("Stim Start", "Stim Start LPC1-LPC2",
                    "Stim Start-LPC1", "Stim Stop",
                    "Stim Stop LOF6-LOF7", "Stim Stop-target"):
        assert wl.matches(phrase, site_code="A"), (
            f"shipped Stim regex should match {phrase!r}"
        )
    # But NOT arbitrary suffixes that go beyond one token — otherwise
    # 'Stim Start CAROL VISITED' would silence the Carol PHI.
    assert not wl.matches("Stim Start CAROL VISITED", site_code="A")

    # And crucially, those same phrases must NOT silence a longer
    # annotation that happens to contain them (fullmatch semantics).
    assert not wl.matches("CAL IN CAROL AT 3PM", site_code="A")
    # Per-site scoping — CUDA phrases don't fire at other sites.
    assert not wl.matches("CAL IN", site_code="S")


def test_matches_uses_fullmatch_not_substring():
    """Fullmatch is what makes pre-filtering safe: unanchored patterns
    can only silence annotations that they cover completely, not any
    annotation that contains them as a substring."""
    import json as _json
    import re as _re
    wl = BoilerplateWhitelist(
        shared=[_re.compile("hello world")],
        per_site={},
    )
    # Exact match → silenced.
    assert wl.matches("hello world")
    # Substring only → NOT silenced (would silence PHI in
    # "hello world dr smith visited" otherwise).
    assert not wl.matches("hello world dr smith visited")
    assert not wl.matches("say hello world!")


# ---------------------------------------------------------------------------
# Delete bucket: 'these should be REMOVED from the EDF, not just
# hidden from the review block'
# ---------------------------------------------------------------------------

def test_delete_bucket_loads_from_shared_and_per_site(tmp_path):
    """POSITIVE: delete_shared / delete_per_site round-trip through
    the loader with the same shape as the whitelist buckets."""
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [],
        "per_site": {},
        "delete_shared": [r"GLOBAL DEBUG.*"],
        "delete_per_site": {"J": [r"Segment: REC START.*"]},
    }))
    wl = load_whitelist(path)
    assert wl.matches_delete("GLOBAL DEBUG foo") is True
    assert wl.matches_delete("Segment: REC START at 10:00",
                              site_code="J") is True
    # Same string with a different site -> only shared applies
    assert wl.matches_delete("Segment: REC START at 10:00",
                              site_code="A") is False


def test_delete_bucket_semantics_independent_of_whitelist(tmp_path):
    """A pattern in the DELETE bucket does NOT also match the
    whitelist (they're separate buckets with separate meaning).
    Regression guard against a merge that conflates them."""
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [], "per_site": {},
        "delete_shared": [r"deletable"],
        "delete_per_site": {},
    }))
    wl = load_whitelist(path)
    assert wl.matches_delete("deletable") is True
    assert wl.matches("deletable") is False


def test_delete_bucket_optional_in_json(tmp_path):
    """Backwards compat: existing whitelist files without
    delete_shared / delete_per_site keys still load. matches_delete
    returns False (nothing to delete)."""
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [r"boilerplate"],
        "per_site": {},
    }))
    wl = load_whitelist(path)
    assert wl.matches("boilerplate") is True
    assert wl.matches_delete("anything") is False


def test_delete_bucket_malformed_type_raises(tmp_path):
    """delete_shared with wrong type should fail LOUDLY at load,
    same as the whitelist buckets. Otherwise a bad file could
    silently disable deletion."""
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({
        "shared": [], "per_site": {},
        "delete_shared": "not a list",
    }))
    with pytest.raises(BoilerplateWhitelistError,
                        match="delete_shared"):
        load_whitelist(path)
