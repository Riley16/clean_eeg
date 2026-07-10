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


_LLM_OP_NAMES: frozenset[str] = frozenset({
    "llm_scan", "llm_date_scan", "llm_name_scan",
})


def _schema_uses_llm_ops(schema: Schema) -> bool:
    for fs in schema.fields.values():
        for op in fs.operations:
            if op.name in _LLM_OP_NAMES:
                return True
    return False


def _apply_auto_apply_llm(schema_raw: dict) -> None:
    """When --auto-apply-llm is set, force report_only=False on every llm_*
    operation in the schema. Modifies `schema_raw` in place."""
    for _fname, fspec in schema_raw.get("fields", {}).items():
        ops = fspec.get("operations")
        if not isinstance(ops, list):
            continue
        for i, op in enumerate(ops):
            if isinstance(op, str) and op in _LLM_OP_NAMES:
                ops[i] = {"name": op, "params": {"report_only": False}}
            elif isinstance(op, dict) and op.get("name") in _LLM_OP_NAMES:
                op.setdefault("params", {})["report_only"] = False


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
    # --- LLM ---
    r.add_argument("--llm-config", default=None,
                   help="Path to llm_config.json. Required when the schema "
                        "uses any llm_* operation.")
    r.add_argument("--llm-server-url", default=None,
                   help="Override server_url from the config.")
    r.add_argument("--llm-model", default=None,
                   help="Override the concrete model resolved for the "
                        "'phi_detector' hint.")
    r.add_argument("--review-out", default=None,
                   help="Where to write the human-reviewable LLM findings "
                        "report. Default: <output>.llm_review.json")
    r.add_argument("--auto-apply-llm", action="store_true", default=False,
                   help="Apply LLM-recommended redactions automatically. "
                        "Off by default — LLM findings are report-only.")
    r.add_argument("--enable-record-review", action="store_true", default=False,
                   help="Run the post-scan per-record LLM review pass.")
    r.add_argument("--llm-cache-clear", action="store_true", default=False,
                   help="Wipe the LLM response cache before running.")

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
        schema_raw = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        if args.auto_apply_llm:
            _apply_auto_apply_llm(schema_raw)
        schema = Schema.from_dict(schema_raw)
    else:
        schema = fmt.default_schema()
        if schema is None:
            print(f"{fmt.name}: --schema is required (no default schema for "
                  f"this format).", file=sys.stderr)
            return 2

    # LLM resources setup — must happen before schema→records processing.
    llm_client = None
    llm_cache = None
    prompt_registry = None
    review_report = None
    llm_model_used: str | None = None
    if _schema_uses_llm_ops(schema):
        if not args.llm_config:
            print("Schema uses llm_* operations but --llm-config was not "
                  "supplied. Point at a JSON config with server_url + models.",
                  file=sys.stderr)
            return 2
        from .llm.cache import LLMCache
        from .llm.client import LLMClient
        from .llm.config import LLMConfig
        from .llm.review_report import ReviewReport
        from .llm.template import PromptRegistry
        try:
            cfg = LLMConfig.load(args.llm_config)
        except (ValueError, IOError) as e:
            print(f"failed to load llm config: {e}", file=sys.stderr)
            return 2
        if args.llm_server_url:
            cfg = cfg.__class__(
                server_type=cfg.server_type,
                server_url=args.llm_server_url.rstrip("/"),
                models=cfg.models, cache_path=cfg.cache_path,
                seed=cfg.seed, temperature=cfg.temperature,
                timeout_seconds=cfg.timeout_seconds,
                max_retries=cfg.max_retries, api_key=cfg.api_key,
            )
        if args.llm_model:
            models = dict(cfg.models)
            models["phi_detector"] = args.llm_model
            cfg = cfg.__class__(
                server_type=cfg.server_type, server_url=cfg.server_url,
                models=models, cache_path=cfg.cache_path,
                seed=cfg.seed, temperature=cfg.temperature,
                timeout_seconds=cfg.timeout_seconds,
                max_retries=cfg.max_retries, api_key=cfg.api_key,
            )
        llm_client = LLMClient(cfg)
        llm_model_used = cfg.resolve_model("phi_detector")
        if cfg.cache_path is not None:
            llm_cache = LLMCache(cfg.cache_path)
            if args.llm_cache_clear:
                n = llm_cache.clear()
                print(f"cleared {n} entries from llm cache "
                      f"{cfg.cache_path}", file=sys.stderr)
        prompt_root = Path(__file__).parent / "llm" / "prompts"
        prompt_registry = PromptRegistry(prompt_root)
        review_out = args.review_out or f"{args.output}.llm_review.json"
        review_report = ReviewReport(
            path=review_out, source=in_path, schema_sha256=schema.sha256(),
        )

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
        llm_client=llm_client,
        llm_cache=llm_cache,
        prompt_registry=prompt_registry,
        review_report=review_report,
    )
    try:
        redacted_records, events = rr.process_records(records)

        if args.enable_record_review:
            if llm_client is None or prompt_registry is None or review_report is None:
                print("--enable-record-review requires --llm-config.",
                      file=sys.stderr)
                return 2
            from .llm.record_reviewer import RecordReviewer
            reviewer = RecordReviewer(
                client=llm_client,
                prompt_registry=prompt_registry,
                review_report=review_report,
                cache=llm_cache,
            )
            reviewer.review_records(redacted_records, schema)
    finally:
        if review_report is not None:
            review_report.close()
        if llm_cache is not None:
            llm_cache.close()
        if llm_client is not None:
            llm_client.close()

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

    if llm_model_used:
        print(f"llm_model used: {llm_model_used}", file=sys.stderr)

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
