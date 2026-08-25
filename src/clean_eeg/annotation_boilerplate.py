"""Per-site regex whitelist for standard/boilerplate EDF annotations.

The de-identifier still redacts these annotations — the whitelist only
controls whether a given redaction is echoed into the end-of-run
'Human review needed' block. Recurring site-specific boilerplate that
happens to contain what looks like PHI (e.g., a technician's initials
in a nightly system-status marker) fires a redaction event every night
across every subject; surfacing every one drowns out the genuinely
novel annotations that actually deserve human review.

Whitelist file format::

    {
      "shared": ["<regex>", ...],
      "per_site": {"S": ["<regex>", ...], "A": [...], ...}
    }

Lookup is keyed by ``subject_code[-1]`` — same convention as
``SITE_CODE_TO_INCOMING_FOLDER``. An unknown site letter falls through
to the shared bucket only (never crashes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BoilerplateWhitelist:
    """Compiled matcher for two families of patterns:

      * ``shared`` / ``per_site``: WHITELIST -- 'known-safe boilerplate,
        silence from the human-review block but KEEP in the EDF output.'
      * ``delete_shared`` / ``delete_per_site``: DELETE -- 'junk that
        should be removed from the EDF output entirely.' Same
        fullmatch semantics; different downstream action.

    All four lists are pre-compiled at load time so each match check
    costs O(pattern count) regex matches, not O(pattern count)
    compilations.
    """

    shared: list[re.Pattern] = field(default_factory=list)
    per_site: dict[str, list[re.Pattern]] = field(default_factory=dict)
    delete_shared: list[re.Pattern] = field(default_factory=list)
    delete_per_site: dict[str, list[re.Pattern]] = field(default_factory=dict)

    def matches(self, text: str, site_code: str | None = None) -> bool:
        """True if ``text`` matches (via ``re.fullmatch``) any shared
        or per-site WHITELIST regex. Full-match semantics -- the
        regex must match the ENTIRE text, not a substring.

        Also matches after stripping an optional leading ``*`` or
        ``*<space>`` prefix. Clinical annotation exports routinely
        prepend a literal asterisk to boilerplate ('*Mark',
        '* RESET OFF') so requiring patterns to explicitly cover
        that prefix would double every JSON entry. Handling it here
        keeps the JSON clean.
        """
        return (self._matches_in(text, site_code,
                                  self.shared, self.per_site)
                or self._matches_in(
                    _strip_asterisk_prefix(text), site_code,
                    self.shared, self.per_site))

    def matches_delete(self, text: str,
                        site_code: str | None = None) -> bool:
        """True if ``text`` matches (via ``re.fullmatch``) any shared
        or per-site DELETE regex. Same asterisk-prefix handling as
        :meth:`matches`. Callers that only want to exclude
        annotations from a review count should use
        ``matches`` OR ``matches_delete``. Callers that will actually
        mutate the EDF (delete these annotations) should use
        ``matches_delete`` alone."""
        return (self._matches_in(text, site_code,
                                  self.delete_shared,
                                  self.delete_per_site)
                or self._matches_in(
                    _strip_asterisk_prefix(text), site_code,
                    self.delete_shared, self.delete_per_site))

    def _matches_in(self, text: str, site_code: str | None,
                     shared: list[re.Pattern],
                     per_site: dict[str, list[re.Pattern]]) -> bool:
        for pat in shared:
            if pat.fullmatch(text):
                return True
        if site_code:
            for pat in per_site.get(site_code, ()):
                if pat.fullmatch(text):
                    return True
        return False


def _strip_asterisk_prefix(text: str) -> str:
    """Strip a leading ``*`` or ``*<whitespace>`` from ``text``.
    Returns the original text unchanged if no such prefix exists.
    Used by :class:`BoilerplateWhitelist` to make the ``*`` prefix
    that clinical exports routinely prepend to boilerplate
    ('*Mark', '* RESET OFF') an implicit optional prefix on every
    whitelist / delete pattern -- so operators don't have to
    duplicate every entry with a ``\\*\\s?`` variant."""
    if text.startswith("* "):
        return text[2:]
    if text.startswith("*"):
        return text[1:]
    return text


class BoilerplateWhitelistError(ValueError):
    """Raised on a malformed whitelist file. We prefer failing loudly at
    load time over silently reverting to 'flag everything', which would
    re-introduce the review-block flood the whitelist exists to prevent."""


def load_whitelist(path: str | Path | None) -> BoilerplateWhitelist:
    """Parse a whitelist JSON file. Missing file → empty whitelist (no
    crash, no boilerplate suppression — the review block will simply
    surface every redaction, which is the safe fallback). Malformed
    JSON or wrong shape → :class:`BoilerplateWhitelistError`.
    """
    if path is None:
        return BoilerplateWhitelist()
    p = Path(path)
    if not p.exists():
        return BoilerplateWhitelist()
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise BoilerplateWhitelistError(
            f"{p}: not valid JSON — {e}"
        ) from e
    if not isinstance(data, dict):
        raise BoilerplateWhitelistError(
            f"{p}: top-level must be an object with 'shared' and "
            f"'per_site' keys, got {type(data).__name__}"
        )
    shared_raw = data.get("shared", [])
    per_site_raw = data.get("per_site", {})
    delete_shared_raw = data.get("delete_shared", [])
    delete_per_site_raw = data.get("delete_per_site", {})
    for name, raw in (("shared", shared_raw),
                        ("delete_shared", delete_shared_raw)):
        if not isinstance(raw, list):
            raise BoilerplateWhitelistError(
                f"{p}: {name!r} must be a list of regex strings")
    for name, raw in (("per_site", per_site_raw),
                        ("delete_per_site", delete_per_site_raw)):
        if not isinstance(raw, dict):
            raise BoilerplateWhitelistError(
                f"{p}: {name!r} must be an object keyed by site code")
    try:
        shared = [re.compile(pat) for pat in shared_raw]
        per_site = {
            site: [re.compile(pat) for pat in patterns]
            for site, patterns in per_site_raw.items()
        }
        delete_shared = [re.compile(pat) for pat in delete_shared_raw]
        delete_per_site = {
            site: [re.compile(pat) for pat in patterns]
            for site, patterns in delete_per_site_raw.items()
        }
    except re.error as e:
        raise BoilerplateWhitelistError(
            f"{p}: invalid regex — {e}"
        ) from e
    return BoilerplateWhitelist(
        shared=shared, per_site=per_site,
        delete_shared=delete_shared, delete_per_site=delete_per_site)
