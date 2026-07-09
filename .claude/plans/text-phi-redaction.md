# Plan (v2): Schema-driven records model for text-PHI redaction

## Context

**Where we are.** Branch `text-phi-redaction` currently ships a working, tested (135 passing, 98% coverage) redaction toolchain at `scripts/text_phi/`. It supports `.txt` and CSV via a pluggable formats registry, `--fmt-opt KEY=VALUE`, post-write validation, and default field mappings for CSV. The redactor consumes raw strings and returns `RedactionResult(text, spans)`.

**Why we're re-architecting.** The current design has no unified internal data model — each format module extracts `(location, text)` pairs ad-hoc, calls the redactor with strings, and writes back. This breaks down when we need to:

1. Redact **dates** (including HIPAA-style shift-to-stay-start), which requires cross-field context (all dates in a record shift by a per-record offset derived from an anchor field).
2. Handle **different internal formats of the same file type** (institution A's CSV has columns `[patient_mrn, note_text]`; institution B's has `[mrn, body]`).
3. Guarantee **lossless roundtrip** across all standard dtypes for the no-op / passthrough case — a precondition for treating this as a trustable redaction tool.
4. Add new file formats (JSON, XLSX, DICOM headers) without duplicating the redaction loop.

The re-architecture standardizes on a **records-of-fields internal representation** driven by an **external JSON schema** that declares each field's dtype, description, and (optional) redaction operations. Adding a new format = write `.load`/`.save`; the redaction loop is written once and shared.

## Design decisions (locked)

- **`operations` key** defaults to sentinel string `"default"` (resolves to the dtype's default operation list). Explicit list fully **replaces** the default. `null` disallowed. Name simplified from earlier `non_default_operations`.
- **`depends_on: {role: field_name, ...}`** — named-mapping form. Each operation declares which roles it consumes; schema validation checks required roles are supplied.
- **CLI**: always explicit `--schema PATH`. Bootstrap via `text_phi schema derive`.
- **Unknown/missing fields**: default `error`. Opt-in `--allow-unknown` demotes to warning + passthrough.
- **Dtype coercion**: eager by default (parse each record at load; fail fast). `--allow-parse-errors` demotes to warning + skip.
- **Operations declare `allowed_dtypes`**. A `date_shift` referenced from a `dtype: string` field is a schema misconfiguration, caught at schema-validate time before any file I/O.
- **Lossless roundtrip**: unit-tested as a matrix over `(format × dtype × value_class)` × `(operations: [], operations: [passthrough])`. Both cases must produce byte-identical output.

## Schema shape (v1)

```json
{
  "schema_version": "1",
  "format": "csv",
  "unknown_field_policy": "error",
  "missing_field_policy": "error",
  "fields": {
    "patient_name": {
      "dtype": "subject_name",
      "description": "Patient full name — used as per-record subject",
      "operations": "default"
    },
    "admission_date": {
      "dtype": "date",
      "description": "Stay admission date; anchor for all other dates",
      "operations": ["date_shift_to_base"]
    },
    "note_date": {
      "dtype": "date",
      "description": "Date the clinical note was written",
      "operations": ["date_shift_relative_to_stay_start"],
      "depends_on": {"stay_start_field": "admission_date"}
    },
    "note_text": {
      "dtype": "string",
      "description": "Free-form clinical note body",
      "operations": "default",
      "depends_on": {"subject_name_field": "patient_name"}
    },
    "channel": {
      "dtype": "string",
      "description": "EEG electrode label (not PHI)",
      "operations": ["passthrough"]
    }
  }
}
```

## Dtype vocabulary (initial)

`string` · `integer` · `float` · `boolean` · `enum` · `date` · `datetime` · `subject_name` · `zip_code` · `phone` · `email` · `ssn` · `mrn` · `url` · `ip` · `bytes`

Each maps to a `DEFAULT_<DTYPE>_REDACT_OPERATIONS` list at module load. Extending the toolkit's default behavior = editing that list, not every schema.

## Operations (initial set)

| Operation | Roles required | Dtypes |
|---|---|---|
| `passthrough` | — | any |
| `constant_replace(value)` | — | any |
| `hash_field` | — | any |
| `subject_name_scan` | `subject_name_field` (optional) | string |
| `generic_phi_scan` | — | string |
| `whitelist_person_filter` | — | string (post-filter on PERSON hits) |
| `parse_subject_name` | — | subject_name — parses into `RecordContext.subject`, then leaves value for a downstream `constant_replace` |
| `date_shift_to_base(base_date)` | — | date, datetime |
| `date_shift_relative_to_stay_start` | `stay_start_field` | date, datetime |
| `date_year_only` | — | date, datetime |
| `date_redact_full` | — | date, datetime |
| `zip_redact` | — | zip_code |
| `phone_redact` / `email_redact` / `ssn_redact` / `url_redact` / `ip_redact` | — | matching dtype |

Each operation is a small class registered in `OPERATIONS`. Adding one is a single-file change plus one entry.

## Directory layout after refactor

```
scripts/text_phi/
  schema.py               # Schema, FieldSpec, load/validate/derive
  dtypes.py               # dtype vocabulary + parse/format helpers
  records.py              # Record, RecordContext, RecordRedactor.process()
  operations/
    __init__.py           # OPERATIONS registry + DEFAULT_<DTYPE>_REDACT_OPERATIONS
    base.py               # Operation Protocol
    passthrough.py        # passthrough, constant_replace, hash_field
    phi_scan.py           # subject_name_scan, generic_phi_scan, whitelist_person_filter, parse_subject_name
    dates.py              # date_shift_to_base, date_shift_relative_to_stay_start, date_year_only, date_redact_full
    typed_phi.py          # zip_redact, phone_redact, email_redact, ssn_redact, url_redact, ip_redact
  formats/
    base.py               # FileFormat protocol: .load(path, schema) -> list[Record]; .save(path, records, schema)
    txt.py                # synthetic single-field records
    csv.py                # df.to_dict(orient="records") load; pd.DataFrame(records).to_csv() save
  audit.py                # references schema_sha256; per-span field + operation
  detectors.py            # unchanged — still consumed by phi_scan operations
  name_parse.py           # unchanged — still consumed by parse_subject_name
  redactor.py             # keeps TextRedactor (low-level); records-model built on top
  cli.py                  # --schema, --allow-unknown, --allow-parse-errors, `schema derive` subcommand
tests/text_phi/
  test_schema.py
  test_dtypes.py
  test_operations/
    test_passthrough.py
    test_phi_scan.py
    test_dates.py
    test_typed_phi.py
  test_records.py             # scheduler + dep resolution + cycle detection
  test_roundtrip_matrix.py    # parameterized (format × dtype × value_class)
  test_csv_format.py          # updated: schema-driven load/save
  test_txt_format.py          # kept: synthetic schema
  test_cli.py                 # updated: --schema; schema derive subcommand
  test_schema_derive.py
```

Files reused unchanged: [`SubjectNameRedactor`](../../src/clean_eeg/anonymize.py#L334), [`PersonalName`](../../src/clean_eeg/anonymize.py#L234), [`whitelist.load_whitelist`](../../src/clean_eeg/whitelist.py#L8-L20), [`whitelist.token_in_whitelist`](../../src/clean_eeg/whitelist.py#L24-L32).

## Migration plan (order matters — each phase keeps tests green)

Every phase runs `pytest tests/text_phi/ --cov=scripts/text_phi` before moving on.

### Phase A — Core primitives (parallel to existing code)
Add without modifying: `dtypes.py`, `operations/base.py`, `operations/{passthrough,phi_scan,dates,typed_phi}.py`, `operations/__init__.py` (registry + `DEFAULT_<DTYPE>_REDACT_OPERATIONS`), `schema.py`.

Tests: `test_dtypes.py`, `test_operations/*`, `test_schema.py`. Existing suite untouched, still green.

### Phase B — Records model
Add `records.py` with `Record`, `RecordContext`, `RecordRedactor.process_records(records, schema, redactor)`. Includes:
- Topological sort over the field dependency graph (cycle → error).
- Per-record context cache (parsed subject name, stay-start offset).
- Preserved-field byte-identity enforcement — any field resolving to a no-op sequence must be byte-identical in the output records; violations raise at end of processing.

Tests: `test_records.py`.

### Phase C — Format refactor
Update `formats/base.py` protocol to `.load(path, schema) -> list[Record]` + `.save(path, records, schema)`. Rewrite `formats/csv.py` around `df.to_dict(orient="records")` / `pd.DataFrame(records).to_csv()`. Rewrite `formats/txt.py` with synthetic single-field records (each line → `{"text": line}`).

Move CSV's post-write structural validation (shape, column list) into the format; the preserved-field check moves to `RecordRedactor`.

Update: `test_csv_format.py`, `test_txt_format.py`, `test_formats_registry.py`.

### Phase D — CLI wiring
Update `cli.py`:
- Add `--schema PATH` (required for redact when the format demands a schema — TXT can be schema-less via a built-in default).
- Add `--allow-unknown`, `--allow-parse-errors`.
- Add subcommand `text_phi schema derive --input PATH --output PATH` — reads the file, infers dtypes, emits a schema with `TODO: verify` markers on every guessed field.
- Retire `--fmt-opt text_columns=…` and `--fmt-opt name_column=…` — those roles now live in the schema.

Update: `test_cli.py`, add `test_schema_derive.py`.

### Phase E — Audit report
Update `audit.py` to embed `schema_sha256`; each span gains `field` + `operation` keys alongside the existing `entity_type` and `recognizer`. Backward-compatible schema evolution (additive fields).

Update: `test_audit.py`.

### Phase F — Roundtrip matrix
Add `test_roundtrip_matrix.py` — pytest-parameterized over:

- Formats: `["csv", "txt"]`
- Dtypes: `["string", "integer", "float", "boolean", "date", "datetime", "enum", "bytes"]`
- Value classes per dtype (empty, edge, extreme, unicode, etc. — full matrix below)
- Schema variants: `operations: []` (pure I/O) and `operations: ["passthrough"]` (redactor loop invoked, nothing changes)

Assertion for both variants: `input.read_bytes() == output.read_bytes()` when the schema is truly no-op. Semantic equality otherwise.

**Value classes per dtype** (representative):

| Dtype | Values |
|---|---|
| string | empty, single ASCII, embedded `\n` and `,` and `"`, leading/trailing whitespace, emoji, high-plane Unicode, NUL char |
| integer | 0, -1, `2**63 - 1`, `-(2**63)`, arbitrary-precision digit-string |
| float | 0.0, -0.0, NaN, +Inf, -Inf, subnormal, 17-sig-fig value |
| boolean | true, false (parse convention per schema) |
| date | ISO, US, European; leap year; DST boundary date |
| datetime | with and without timezone |
| enum | each declared level |
| bytes | empty, all-zero, arbitrary binary (base64 in JSON, base64 in CSV) |
| — | null vs empty-string vs missing (three distinct outcomes preserved) |

### Phase G — Detectors module
No changes to `detectors.py` or `name_parse.py`. They remain the implementation backing `generic_phi_scan` and `parse_subject_name` respectively.

## Test strategy

- Every operation: positive (doesn't fire on non-matching values) and negative (fires on every matching value + common variants).
- Schema validation: every reject-reason has a test (unknown dtype, unregistered operation, missing dep role, cycle in deps, dtype-incompatible operation).
- Roundtrip matrix as above.
- Coverage target ≥ 95% for `scripts/text_phi/`.
- Not retested (still trusted upstream): `src/clean_eeg/anonymize.py` internals, Presidio recognizers, `whitelist.load_whitelist` file I/O.

## End-to-end verification

1. `"$CONDA_ENV_PATH/bin/pytest" tests/text_phi/ --cov=scripts/text_phi --cov-report=term-missing` — coverage ≥ 95%, all cases green.
2. `"$CONDA_ENV_PATH/bin/pytest"` — full existing suite still green (no `src/clean_eeg/` changes).
3. Bootstrap smoke: create a small `clinical_notes.csv` with `patient_name`, `admission_date`, `note_date`, `note_text`, `channel`. Run `text_phi schema derive --input clinical_notes.csv --output notes.schema.json`. Manually edit dtypes for `admission_date` / `note_date` / `patient_name`. Add `depends_on` for `note_date` and `note_text`. Run `text_phi redact --input clinical_notes.csv --schema notes.schema.json --output redacted.csv --audit-out audit.json`. Verify: patient name gone, dates shifted with interval preserved (`note_date - admission_date` unchanged), `channel` byte-identical.
4. Lossless smoke: write a CSV with every dtype's edge-case value. Use a schema with `operations: []` everywhere. Round-trip and byte-compare.

## Out of scope for this plan (explicit)

- LLM redactor (still deferred; seam preserved — LLM ops slot into the operations registry).
- EDF pipeline changes.
- Iterator-mode format loaders for streaming huge files (deferred; user OK with in-memory for now).
- Schema auto-detection from file header (opt-in future feature; explicit `--schema` for now).
- Format-specific dtypes beyond the initial vocabulary above (e.g. DICOM sequence tags) — added when their format modules are added.
- Merging `text-phi-redaction` into `main` — decided after review.
