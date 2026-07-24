"""Environment provenance for ``log.out``.

Records what code produced a de-identified subject: clean_eeg's own
version + git commit, the raw command line the operator invoked, the
Python and OS versions in effect, and the versions of every load-bearing
dependency. When something goes wrong months later (a dep regression,
a bug in a commit that had a subtle bug, a system-python change), the
recorded provenance is what lets a maintainer reproduce the exact stack.

Called from ``clean_subject_eeg.py`` **after** PHI patterns have been
registered with the ``PipelineLogger`` — the tee's on-write scrubbing
then masks any PHI in the command line (name arguments the operator
typed).

Safety-by-construction: nothing recorded here identifies the operator
or their environment beyond kernel/arch — hostname, username, and env
vars are deliberately excluded.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clean_eeg.log import PipelineLogger


# Key runtime dependencies. When a bug turns out to be a dep regression
# the maintainer needs to know exactly which upstream version the
# operator was running — a wheel from 2 years ago picks up different
# transitive versions than a fresh install.
KEY_DEPENDENCIES: tuple[str, ...] = (
    "pyedflib",
    "edfio",
    "lunapi",
    "mne",
    "presidio-analyzer",
    "presidio-anonymizer",
    "rapidfuzz",
    "nicknames",
    "numpy",
    "spacy",
)


def _git_provenance() -> tuple[str | None, bool]:
    """Return ``(commit_sha, is_dirty)`` for the clean_eeg source tree,
    or ``(None, False)`` if this isn't a git checkout / git is not
    available. Runs both queries with a short timeout so a hung git
    process on a slow FS can never block de-identification.
    """
    import clean_eeg
    pkg_dir = Path(clean_eeg.__file__).resolve().parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(pkg_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(pkg_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None, False


def _package_version(pkg_name: str) -> str:
    try:
        return _pkg_version(pkg_name)
    except PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"error: {type(exc).__name__}"


def build_provenance_block(*,
                           argv: list[str] | None = None,
                           git_provenance=None,
                           dependencies: tuple[str, ...] = KEY_DEPENDENCIES,
                           ) -> str:
    """Assemble the provenance text block. Pure — no I/O, no logger.
    Injectable ``git_provenance`` and ``argv`` so tests can control the
    reported values without mocking subprocess or sys.argv.
    """
    argv = argv if argv is not None else sys.argv
    sha, dirty = (git_provenance or _git_provenance)()

    lines = ["\n=== Provenance ==="]
    lines.append(f"  clean_eeg version: {_package_version('clean_eeg')}")
    if sha is not None:
        marker = "  DIRTY (uncommitted edits at run time)" if dirty else ""
        lines.append(f"  clean_eeg commit:  {sha}{marker}")
    else:
        lines.append("  clean_eeg commit:  unknown (not a git checkout)")
    lines.append(f"  command:           {' '.join(argv)}")
    lines.append(f"  python:            {sys.version.split()[0]}")
    lines.append(f"  platform:          "
                 f"{platform.system()} {platform.release()} ({platform.machine()})")
    lines.append("  dependencies:")
    for pkg in dependencies:
        lines.append(f"    {pkg}: {_package_version(pkg)}")
    lines.append("==================\n")
    return "\n".join(lines) + "\n"


def log_environment_provenance(logger: "PipelineLogger") -> None:
    """Write the provenance block to ``logger``. PHI in ``sys.argv`` is
    scrubbed on write by the tee — the caller must register PHI patterns
    via ``logger.add_phi`` before calling this."""
    logger.write_to_log(build_provenance_block())
