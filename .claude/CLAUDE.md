# clean_eeg - Project Instructions

## Project Overview
Python package for de-identifying (removing PHI from) clinical EEG data stored in European Data Format (EDF) files. Developed at the Computational Memory Lab at the University of Pennsylvania for intracranial EEG research. Primarily tested against Nihon Kohden EDF exports.

## Git Conventions
- Use **conventional commits** style for all git commits (e.g., `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`)
- Don't include "Claude co-authorship" statements in commit messages.
- Version is derived from git tags via setuptools-scm (format: `v1.2.3`)

## Project Structure
```
src/clean_eeg/          # Main package source
  clean_subject_eeg.py  # Main pipeline orchestrator + CLI entry point
  anonymize.py          # Name redaction (Presidio + fuzzy matching + nicknames)
  modify_edf_inplace.py # Binary-level in-place EDF header/annotation editing
  split_discontinuous_edf.py  # EDF+D → EDF+C splitting via Luna
  load_eeg.py           # EDF I/O abstraction (pyedflib, edfio, MNE, lunapi backends)
  compare_eeg.py        # EDF file comparison/validation
  whitelist.py          # Word whitelists for false-positive name filtering
  paths.py              # Central path configuration
scripts/                # CLI wrapper scripts and utilities
  build_whitelist.py    # Build word whitelists from SUBTLEX-US frequency data
  anonymize_csv.py      # Scan CSVs for potential PII
  clean_subject_eeg.py  # CLI wrapper (thin)
tests/                  # pytest test suite
  config.json           # Test EDF generation configurations
  generate_edf.py       # Synthetic test EDF file generation
  conftest.py           # Session-scoped fixture auto-generates test data
data/                   # Whitelists, name datasets (gitignored)
```

## Build & Test
- Python >=3.11, install with `pip install -e .`
- Run tests: `pytest` from project root
- Test data is auto-generated on first test run via session-scoped conftest fixture
- Test data and `data/` directories are gitignored

## Key Dependencies
| Library | Role |
|---------|------|
| pyedflib | Primary EDF read/write and header manipulation |
| edfio | Annotation handling, alternative EDF I/O |
| lunapi | Luna interface for discontinuous EDF splitting/segment detection |
| MNE | Alternative EDF loading backend |
| presidio-analyzer/anonymizer | NLP entity detection and redaction framework |
| rapidfuzz | Levenshtein distance for fuzzy name matching |
| nicknames | Nickname variant expansion (e.g., John → Johnny) |

## Architecture Notes

### De-identification Pipeline (clean_subject_eeg.py)
1. Load all EDF files from input directory
2. Validate metadata consistency (names, signal headers, recording gaps)
3. Compute earliest recording start time
4. For each file: redact header fields, redact annotations, write output
5. Optional in-place mode: modify headers in original file + create separate annotation stub

### Name Redaction (anonymize.py)
Three-layer approach using Presidio:
1. **Exact match** — deny-list with punctuation/possessive variants
2. **Title + initials patterns** — regex for "Dr. J. O'Connor" style
3. **Fuzzy matching** — Levenshtein distance ≤1 via custom recognizer

### In-Place EDF Modification (modify_edf_inplace.py)
- Creates temp EDF with updated headers via pyedflib
- Copies only header bytes back to original file (preserves signal data)
- Uses precise byte offsets per EDF spec (main header: 256 bytes, signal headers: 256 bytes each)

### EDF Format Handling
- Supports EDF, EDF+C (continuous), EDF+D (discontinuous)
- Format detected via reserved field at byte offset 192
- Discontinuous files split into continuous segments via Luna before processing

## Important Constants
- `BASE_START_DATE = datetime(1985, 1, 1)` — reference date for de-identified timestamps
- `SUBJECT_CODE_PATTERN = r'^R1\d{3}[ACDEFHJMNPST]$'` — expected subject code format
- `MAX_RECORDING_GAP_SECONDS = 60` — max allowed gap between consecutive recordings

## CLI Usage
```
python src/clean_eeg/clean_subject_eeg.py \
  --input_path PATH --output_path PATH \
  --subject_code R1XXXY \
  --first-name John --middle-name Paul --last-name Smith
```

## Task Tracking
Current TODOs are tracked in `TODO.md` at the project root (gitignored).

## Environment
Run all code from the conda environment `clean_eeg`. Its absolute path is exposed to Bash tool calls as `$CONDA_ENV_PATH` (set per-machine via the `env` block in `.claude/settings.local.json`). Invoke env binaries directly — e.g. `"$CONDA_ENV_PATH/bin/pytest"` — instead of `conda activate` (which fails non-interactively).
