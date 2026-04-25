# TODO

## updates from call with Aaron Geller
- [ ] extend README conda install instructions, include .bashrc instructions? may want to build a one-shot install script

## In-Place EDF Writing

- [x] Confirm in-place EDF writing covers all header fields
  - All header fields tested (all relevant fields exposed by pyedflib and modified according to config.json)
- [x] Add in-place signal header writing
- [x] ~~Add in-place annotation writing~~ — abandoned in favor of stub EDF approach (see commit `c665f31`)
- [x] Add more unit tests for in-place writing
  - Added: clear annotations standalone, create annotations-only EDF, header-only update, file size preservation, truncation validation
- [x] Integrate in-place writing into de-identification pipeline
  - `validate_header_roundtrip()` captures pyedflib truncation warnings before writing
  - Truncation warnings printed during pipeline execution

## Annotation Stub EDF

- [x] Write function to merge annotation stub EDF back into main data EDF (for use on cluster after transfer)
  - `merge_annotation_stub_edf()` reads annotations from stub, encodes TAL bytes, writes into data EDF's annotation records
- [x] Ensure copying from EDF stub back to original file cannot be interrupted
  - Atomic file replacement via `os.replace()` on a temp copy; original untouched on failure
  - Runtime integrity contract (`_verify_merge_integrity`) checks signals, headers, signal headers, and annotations before committing

## Validation & Logging

- [x] Add log.out that is sent back with EDF files for debugging (must not contain PHI)
- [x] Add random audit to check if de-identification modified EDF signals or non-edited fields
  - Samples up to 2 files per subject, verifies bit-identical signals after in-place write
  - Skips audit for rewrite path (pyedflib digital/physical conversion introduces known floating-point differences)

## mmap signal loader (`_read_signals_via_mmap`) — integrity coverage checklist

Systematic coverage of edge cases that match each pyedflib C-library
integrity check (`EDFLIB_FILE_ERRORS_*` in `edflib.c`). `load_edf` only
reaches the mmap helper when `preload=True, read_digital=True,
use_mmap=True` AND pyedflib successfully opens the file. Any raise
from the helper triggers the pyedflib-readSignal fallback.

### Byte-identical vs pyedflib (matrix tests)
- [x] Single channel (1)
- [x] A handful of channels (2, 8)
- [x] Medium count (16, 50)
- [x] Full NK geometry (178 channels)
- [x] Low sample rates (100, 200, 250 Hz)
- [x] Medium sample rates (500, 1000 Hz)
- [x] High sample rate (2000 Hz)
- [x] NK-short record duration (0.086 s)
- [x] Standard record duration (1.0 s)
- [x] Sub-second record duration (0.5 s)
- [x] Long record duration (2.0 s)
- [x] Heterogeneous sample rates within one file (100/200/500/1000 Hz)
- [x] Short file (3 records)
- [x] Long file (1000 records)
- [x] Minimum-case file (1 channel, 1 record)

### Edge-case error handling (helper called directly)
Our loader's defensive responsibilities beyond pyedflib's open-time
checks:
- [x] `EDFLIB_FILE_ERRORS_FILESIZE` analogue: file smaller than
      header claims → `ValueError("smaller than expected")`
- [x] `EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD` analogue: a signal's
      samples_per_record byte-field contains 0 →
      `ValueError("non-positive")`
- [x] Unparseable samples_per_record bytes →
      `ValueError("Unparseable samples_per_record")`
- [x] `EDFLIB_FILE_ERRORS_NUMBER_SIGNALS` analogue: n_signals = 0 →
      empty list (helper short-circuits)
- [x] `EDFLIB_FILE_ERRORS_NUMBER_DATARECORDS` analogue: n_records = 0 →
      empty list (helper short-circuits)
- [x] File larger than header claims → tolerated (trailing bytes
      ignored, same semantics as pyedflib)

### Fallback contract (load_edf → mmap helper)
- [x] Any exception from the mmap helper triggers a warning and falls
      back to pyedflib's per-channel readSignal loop
- [x] Fallback produces byte-identical int32 output to the helper's
      successful path

### Non-coverage at the mmap helper (pyedflib runs first — documented boundary)

The mmap helper runs *after* `pyedflib.EdfReader(filename)` succeeds.
Any check below fires at pyedflib's open, so the helper itself never
needs to validate them. But the **pipeline overall** has to handle
each one — either by pre-repairing the pathology in
`_load_edf_metadata`, or by letting pyedflib reject the file and
catching that cleanly in the per-file skip path.

#### Proactively repaired in `_load_edf_metadata`
These are real NK export pathologies we've seen and the repair runs
*before* `load_edf` so pyedflib never sees the broken file:
- [x] `EDFLIB_FILE_ERRORS_PHYS_MAX` / `EDFLIB_FILE_ERRORS_PHYS_MIN`
      — degenerate or unparseable phys range —
      `repair_degenerate_signal_ranges` ([src/clean_eeg/repair_edf.py](src/clean_eeg/repair_edf.py))
- [x] `EDFLIB_FILE_ERRORS_DIG_MAX` — degenerate digital range —
      same repair (phys + dig rewritten together as uncalibrated)
- [x] `EDFLIB_FILE_ERRORS_FILESIZE` — truncated file (NK
      mid-session-stop pathology) — `repair_truncated_edf_header`
- [x] EDF+D header on actually-continuous data — `convert_edfC_to_edfD`

#### Handled by skip-with-warning (no proactive repair)
We haven't seen these in real NK exports, so no repair yet. If they
occur, pyedflib's open raises `OSError`, `_load_edf_metadata`'s
per-file try/except catches it, the file lands in the "skipped
files" summary, and the operator is told to send `log.out` to the
data management team. The rest of the subject's files still process.
- [x] Skip-with-warning path exercised by
      `test_pipeline_skips_file_with_non_ascii_label_gracefully`
      (in `tests/test_clean_subject_eeg.py`) — a file with a non-
      printable byte in a signal label is rejected by pyedflib
      (`EDFLIB_FILE_ERRORS_LABEL`), collected into the skipped-files
      summary, and does not crash the run.
- [ ] `EDFLIB_FILE_ERRORS_LABEL` — label contains non-ASCII chars
- [ ] `EDFLIB_FILE_ERRORS_TRANSDUCER` — transducer non-ASCII
- [ ] `EDFLIB_FILE_ERRORS_PREFILTER` — prefilter non-ASCII
- [ ] `EDFLIB_FILE_ERRORS_PHYS_DIMENSION` — dimension non-ASCII
- [ ] `EDFLIB_FILE_ERRORS_RESERVED_FIELD` — reserved field non-ASCII
- [ ] `EDFLIB_FILE_ERRORS_STARTDATE` — day/month out of range
- [ ] `EDFLIB_FILE_ERRORS_STARTTIME` — h/m/s out of range
- [ ] `EDFLIB_FILE_ERRORS_BYTES_HEADER` — header_bytes ≠
      `256 * (n_signals + 1)`
- [ ] `EDFLIB_FILE_ERRORS_RECORDINGFIELD` — recording_id subformat
      violation (missing "Startdate …" prefix, etc.)
- [ ] `EDFLIB_FILE_ERRORS_DURATION` — data_record_duration non-
      numeric or negative
- [ ] `EDFLIB_FILE_CONTAINS_FORMAT_ERRORS` — version/identifier
      mismatch or other generic format error

#### Candidate proactive repairs (if we see them in the wild)
Low-risk fixes we could add alongside the existing repairs in
`repair_edf.py`. Not implemented yet — current behaviour is skip-
with-warning, which is safe but loses the file. If any of these
start showing up frequently in NK exports, promote to proactive
repair.
- [ ] Sanitize non-ASCII bytes in label / transducer / prefilter /
      dimension / reserved → replace with `'?'` or space.
      Descriptive-only fields; no signal-data impact.
- [ ] Invalid startdate → substitute EDF+ clipping date `01.01.85`.
      De-identification already rewrites startdate to a shifted
      time, so the repair would be invisible to the output.
- [ ] Invalid starttime → `00.00.00`.
- [ ] BYTES_HEADER mismatch → recompute from n_signals and rewrite
      the bytes-184-to-192 field. Only safe when n_signals is
      itself parseable and plausible.
- [ ] Non-numeric/negative data_record_duration → defensive
      decision: skip the file (repair requires guessing correct
      duration, which can silently rescale signals).

### Follow-ups worth considering later
- [ ] Property-based / fuzz test: random valid EDF+ fixtures with
      random geometries, assert byte-identical vs pyedflib output
- [ ] Bench the mmap path on the real 3.8 GB NK file to quantify
      minutes → seconds win in situ
- [x] ~~Return int16 instead of int32 for 2x memory savings~~ —
      done in commit `1e7df5e`; downstream consumers (audit via
      np.array_equal, writeSamples via np.issubdtype(integer))
      accept integer dtypes generically.

## Deployment

- [ ] Test install on Windows
- [ ] Run on full Jeff subject
- [ ] Distribute to sites

---

## Later

- [ ] Set up CI (Travis or GitHub Actions)
- [ ] Code linter/black

## Wish List

- [ ] Finish `split_discontinuous_edf.py` functionality
  - Currently using hack of overwriting 'reserved' field of headers of continuous EDF+D files from Nihon Kohden ('EDF+D' → 'EDF+C')
- [ ] Finish explicit check for continuous data
  - Currently relying on lunapi
  - Should implement own method rather than relying on lunapi
    - EDF+ standard is clean/simple; compare with lunapi's methods
    - ~~Try MNE~~ — will not work (MNE ignores record timing TALs, treats EDF+D as EDF+C, does not provide timing annotations)
  - Then use pyedflib for standard conversion
    - Or check whether MNE loads discontinuous EDF+D as a check for discontinuous data
  - Confirm that lunapi does not modify EDF header/signal header info during conversion, except for start time
