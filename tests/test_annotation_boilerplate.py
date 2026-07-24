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
    """The empty shipped whitelist at data/annotation_boilerplate_whitelist.json
    must load cleanly — sites populate it over time."""
    from clean_eeg.paths import ANNOTATION_BOILERPLATE_WHITELIST_PATH
    wl = load_whitelist(ANNOTATION_BOILERPLATE_WHITELIST_PATH)
    # Empty by default; matches nothing.
    assert not wl.matches("anything", site_code="A")
    assert isinstance(wl, BoilerplateWhitelist)
