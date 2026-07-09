"""CLI: schema-driven PHI redaction of a supported file, plus a `schema
derive` subcommand that bootstraps a starting schema from an input file.

Redact:
  python -m scripts.text_phi.cli redact \
    --input in.csv --output out.csv --schema notes.schema.json \
    --mode both [--subject-first NAME --subject-last NAME] \
    [--audit-out audit.json] [--allow-unknown] [--allow-parse-errors]

Derive schema:
  python -m scripts.text_phi.cli schema derive \
    --input notes.csv --output notes.schema.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clean_eeg.anonymize import PersonalName

from .audit import AuditWriter
from .formats import get_format, infer_columns, supported_extensions
from .records import RecordRedactor
from .redactor import TextRedactor
from .schema import Schema, derive_schema_from_columns


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="text_phi",
        description="Schema-driven PHI redaction for text/CSV files.",
    )
    subs = p.add_subparsers(dest="command", required=True)

    # --- redact ---
    r = subs.add_parser("redact", help="Redact PHI from a file.")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--schema", default=None,
                   help="Path to schema JSON. Required for CSV. For TXT the "
                        "format's default schema is used if omitted.")
    r.add_argument("--mode", choices=("subject", "generic", "both"),
                   default="both")
    r.add_argument("--subject-first", default=None)
    r.add_argument("--subject-middle", action="append", default=None)
    r.add_argument("--subject-last", default=None)
    r.add_argument("--allow-unknown", action="store_true", default=False)
    r.add_argument("--allow-parse-errors", action="store_true", default=False)
    # Preprocessing (subject-level field propagation). On by default so the
    # subject_name_scan / date_shift_relative_to_stay_start operations have
    # the values they need on every row.
    r.add_argument("--skip-preprocess", action="store_true", default=False,
                   help="Skip the subject-field propagation pass "
                        "(default: preprocess is on).")
    r.add_argument("--subject-key", default="subject_number",
                   help="Column that identifies a subject across rows "
                        "(default: subject_number)")
    r.add_argument("--propagate", default="subject_name,implant_date",
                   help="Comma-separated columns to fill down within each "
                        "subject before redaction "
                        "(default: subject_name,implant_date)")
    r.add_argument("--enable-zip", dest="enable_zip",
                   action="store_true", default=True)
    r.add_argument("--no-enable-zip", dest="enable_zip", action="store_false")
    r.add_argument("--enable-age", dest="enable_age",
                   action="store_true", default=True)
    r.add_argument("--no-enable-age", dest="enable_age", action="store_false")
    r.add_argument("--mrn-regex", default=None)
    r.add_argument("--audit-out", default=None)
    r.add_argument("--audit-include-original", action="store_true", default=False)
    r.add_argument("--replacement-style",
                   choices=("literal", "labeled"), default="labeled",
                   help="'labeled' replaces spans with e.g. [SUBJECT_NAME] "
                        "(default). 'literal' uses --literal-replacement.")
    r.add_argument("--literal-replacement", default="X")
    r.add_argument("--entities", default=None,
                   help="Comma-separated entity types to keep from the generic layer.")

    # --- schema derive ---
    s = subs.add_parser("schema", help="Schema helpers.")
    s_subs = s.add_subparsers(dest="schema_command", required=True)
    d = s_subs.add_parser("derive", help="Bootstrap a starting schema.")
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)

    return p


def _parse_subject(args) -> PersonalName | None:
    if not (args.subject_first or args.subject_last):
        return None
    return PersonalName(
        first_name=args.subject_first or "",
        middle_names=list(args.subject_middle or []),
        last_name=args.subject_last or "",
    )


def _split_csv_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _cmd_schema_derive(args) -> int:
    in_path = Path(args.input)
    ext = in_path.suffix.lower()
    if ext == ".csv":
        columns = infer_columns(in_path)
    elif ext == ".txt":
        # TxtFormat has a canned schema; no need to derive.
        from .formats import TxtFormat
        schema = TxtFormat().default_schema()
        _write_schema(schema, args.output)
        return 0
    else:
        print(f"Unsupported input extension for derive: {ext!r}",
              file=sys.stderr)
        return 2
    schema = derive_schema_from_columns(columns, format_name=ext.lstrip("."))
    _write_schema(schema, args.output)
    return 0


def _write_schema(schema: Schema, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(schema.raw, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _cmd_redact(args) -> int:
    in_path = Path(args.input)
    ext = in_path.suffix.lower()
    fmt = get_format(ext)
    if fmt is None:
        print(f"Unsupported input extension: {ext!r}. "
              f"Supported: {supported_extensions()}", file=sys.stderr)
        return 2

    schema: Schema | None
    if args.schema:
        schema = Schema.load(args.schema)
    else:
        schema = fmt.default_schema()
        if schema is None:
            print(f"{fmt.name}: --schema is required (no default schema for "
                  f"this format).", file=sys.stderr)
            return 2

    subject = _parse_subject(args)
    entities = _split_csv_list(args.entities)

    text_redactor = TextRedactor(
        mode=args.mode,
        subject_names=[subject] if subject else [],
        replacement_style=args.replacement_style,
        literal_replacement=args.literal_replacement,
        enable_zip=args.enable_zip,
        enable_age=args.enable_age,
        mrn_regex=args.mrn_regex,
        entities=entities,
    )

    # Preprocess: propagate subject-level fields (e.g. subject_name,
    # implant_date) onto every row of the same subject. Enabled by default
    # so per-row operations that depend on those values still work on
    # daily-testing-report rows that don't otherwise carry them.
    preprocessed_tmp: Path | None = None
    load_path = in_path
    if ext == ".csv" and not args.skip_preprocess:
        from .inspect.preprocess_redcap import propagate_subject_fields
        import pandas as pd
        try:
            df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
            propagate_fields = [f.strip() for f in args.propagate.split(",")
                                if f.strip()]
            # Only propagate columns that actually exist; silently skip
            # missing ones (schemas vary across projects).
            propagate_fields = [f for f in propagate_fields if f in df.columns]
            if propagate_fields and args.subject_key in df.columns:
                df = propagate_subject_fields(df, args.subject_key, propagate_fields)
                preprocessed_tmp = Path(args.output).with_suffix(
                    ".preprocessed.csv"
                )
                preprocessed_tmp.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(preprocessed_tmp, index=False)
                load_path = preprocessed_tmp
                print(
                    f"preprocessed: propagated {propagate_fields} across "
                    f"subject '{args.subject_key}' → {preprocessed_tmp}",
                    file=sys.stderr,
                )
        except (ValueError, IOError) as e:
            print(f"preprocess failed: {e}", file=sys.stderr)
            return 2

    try:
        records = fmt.load(
            load_path, schema,
            allow_unknown=args.allow_unknown,
            allow_parse_errors=args.allow_parse_errors,
        )
    except (ValueError, IOError) as e:
        print(str(e), file=sys.stderr)
        return 2

    rr = RecordRedactor(
        schema=schema,
        text_redactor=text_redactor,
        default_subject=subject,
    )
    redacted_records, events = rr.process_records(records)

    fmt.save(args.output, redacted_records, schema)
    fmt.validate_output(
        input_path=in_path,
        output_path=args.output,
        schema=schema,
        n_records=len(redacted_records),
    )

    if args.audit_out:
        subject_names = [subject.get_full_name()] if subject else []
        with AuditWriter(
            path=args.audit_out,
            source_path=in_path,
            mode=args.mode,
            schema=schema,
            replacement_style=args.replacement_style,
            include_original=args.audit_include_original,
            subject_names=subject_names,
        ) as aw:
            for ev in events:
                aw.add_event(ev)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "redact":
        return _cmd_redact(args)
    if args.command == "schema":
        if args.schema_command == "derive":
            return _cmd_schema_derive(args)
    return 2  # unreachable given argparse required=True


if __name__ == "__main__":
    sys.exit(main())
