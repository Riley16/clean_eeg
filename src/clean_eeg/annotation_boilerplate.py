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
    """Compiled matcher. ``shared`` regexes apply to every site;
    ``per_site`` regexes apply only when queried with the matching site
    letter. Both lists are pre-compiled at load time so each redaction
    event costs O(pattern count) regex matches, not O(pattern count)
    compilations."""

    shared: list[re.Pattern] = field(default_factory=list)
    per_site: dict[str, list[re.Pattern]] = field(default_factory=dict)

    def matches(self, text: str, site_code: str | None = None) -> bool:
        """True if ``text`` matches (via ``re.fullmatch``) any shared
        regex, or any regex under the ``site_code`` bucket. Unknown
        ``site_code`` falls through to shared-only. ``None``
        site_code also uses shared-only.

        Full-match semantics — the regex must match the ENTIRE text,
        not a substring — because the intended use is "silence known-
        safe phrases outright." A substring-match on a permissive
        pattern like ``"CAL IN"`` could silence a legitimate PHI-
        bearing annotation like ``"CAL IN CAROL AT 3PM"`` where the
        interesting content follows the boilerplate. Operators who
        want partial matches can write ``".*something.*"`` explicitly.
        """
        for pat in self.shared:
            if pat.fullmatch(text):
                return True
        if site_code:
            for pat in self.per_site.get(site_code, ()):
                if pat.fullmatch(text):
                    return True
        return False


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
    if not isinstance(shared_raw, list):
        raise BoilerplateWhitelistError(
            f"{p}: 'shared' must be a list of regex strings"
        )
    if not isinstance(per_site_raw, dict):
        raise BoilerplateWhitelistError(
            f"{p}: 'per_site' must be an object keyed by site code"
        )
    try:
        shared = [re.compile(pat) for pat in shared_raw]
        per_site = {
            site: [re.compile(pat) for pat in patterns]
            for site, patterns in per_site_raw.items()
        }
    except re.error as e:
        raise BoilerplateWhitelistError(
            f"{p}: invalid regex — {e}"
        ) from e
    return BoilerplateWhitelist(shared=shared, per_site=per_site)
