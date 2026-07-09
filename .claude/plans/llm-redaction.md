# Plan: LLM agent redaction pathway (v2 — local-first, model-agnostic)

## Context

Presidio + our subject-name redactor + REDCap-driven schema handle 95% of PHI. The LLM pass exists for what they miss:

- **Freeform dates** in prose: `"three weeks after implant"`, `"early spring '19"`.
- **Names embedded in prose that don't match the deny-list**: family members, providers.
- **Paraphrased identifiers**: `"the pt from the Colorado group"`, `"her son's college"`.
- **Novel PHI patterns**: implant serial numbers with unusual formats.

### Constraints locked with the user

- **Deployment must fit a laptop first.** The REDCap CSV work happens on the user's local machine (a MacBook, based on the file paths in this repo). We can't assume 4× L40S here — that's for other datasets. Framework must scale up to the cluster later without code changes.
- **Model-agnostic framework.** No model is committed. The framework must let the user swap models in and out easily.
- **The LLM server should be a general-purpose local resource** the user can also use for other tasks in the lab, not a PHI-specific silo.
- **Reproducibility**: seed control + `temperature=0` on every call.
- **Human-reviewable output by default** — LLM operations produce a report of `{original value, recommended redaction, spans, reasons}`; the field is not modified unless the user opts into auto-apply.
- **Accuracy benchmark suite required** — synthetic test set of 25 name-containing + 25 date-containing free-text passages (10 overlap), plus typo/variant perturbations. We measure precision/recall/F1/latency before trusting the LLM in production.

## Serving strategy — OpenAI-compatible everywhere

Every open-source LLM server we'd consider exposes an **OpenAI-compatible chat-completions API**. We write one client against that API and swap the underlying server via config:

| Server | Where | Structured output | Notes |
|---|---|---|---|
| **Ollama** | **local (laptop) — default** | `format: <json_schema>` | Best-in-class UX. `ollama pull qwen2.5:7b`; `ollama serve` on `localhost:11434`. Handles Metal on Apple Silicon, CUDA on Linux. Model swap = one command. |
| **vLLM** | cluster | `extra_body: {guided_json: schema}` | Tensor-parallel across GPUs, continuous batching. For the L40S cluster when we get there. |
| **llama.cpp / LM Studio** | local | JSON via grammar file | Fallback / advanced users. |
| **OpenAI-compatible generally** | any | varies | The client detects `server_type` from config and formats the structured-output body accordingly. |

**Client stays server-agnostic.** Users switch environments by editing `llm_config.json`:

```json
{
  "server_type": "ollama",
  "server_url": "http://localhost:11434/v1",
  "models": {
    "phi_detector":    "qwen2.5:7b-instruct",
    "record_reviewer": "qwen2.5:7b-instruct"
  },
  "cache_path": "temp/llm_cache.sqlite",
  "seed": 42,
  "temperature": 0.0,
  "timeout_seconds": 60,
  "max_retries": 3
}
```

For the cluster: change `server_type: "vllm"`, `server_url: "http://cluster:8000/v1"`, models to `"meta-llama/Llama-3.3-70B-Instruct"`. No code changes.

## Model recommendations for laptop (default)

For a MacBook with ≥16 GB unified memory. All accessed via Ollama tags:

| Model | Size | RAM (~q4) | Notes |
|---|---:|---:|---|
| **Qwen 2.5 7B Instruct** (**recommended default**) | 7B | ~5 GB | Strongest JSON reliability in class; solid at NER-style tasks; runs fine on Metal. |
| **Llama 3.2 3B Instruct** | 3B | ~2.5 GB | Fastest local; good for iteration; may miss subtle PHI. |
| **Qwen 2.5 14B Instruct** | 14B | ~10 GB | Higher accuracy if the laptop has 32+ GB unified. |
| **Phi-4 14B** | 14B | ~9 GB | Competitive alternative to Qwen 14B. |
| **Gemma 3 4B** | 4B | ~3 GB | Fast, capable middle ground. |

Cluster models (same client, just config swap):

| Model | vLLM ID |
|---|---|
| **Llama 3.3 70B Instruct** | `meta-llama/Llama-3.3-70B-Instruct` |
| **Qwen 2.5 72B Instruct** | `Qwen/Qwen2.5-72B-Instruct` |
| **Qwen 3 32B** | `Qwen/Qwen3-32B` |

The redaction operations never reference model names — they only use `model_hint: "phi_detector"` and the config resolves that to a concrete model.

## Reusability — general-purpose LLM client

The client library lives at `scripts/text_phi/llm/client.py` but is written so any lab script can `from scripts.text_phi.llm.client import LLMClient` and get a plain "call a local LLM" helper. No PHI logic in `client.py`, `config.py`, `cache.py`, or `template.py` — they're the general-purpose layer.

PHI-specific code (operations, prompts, review report) lives in sibling modules that build on top.

## Architecture

```
scripts/text_phi/llm/
  __init__.py
  config.py              # LLMConfig + ServerType enum + model registry
  client.py              # Sync httpx OpenAI-compatible client; server-type-aware structured-output body
  cache.py               # SQLite content-hash cache; keyed by (server, model, prompt_hash, input_hash, context_hash, seed)
  template.py            # Jinja rendering: prompt + per-key context templates
  response.py            # Guided JSON schema for spans; matched_text → offset resolution
  review_report.py       # Human-reviewable JSON report writer (per-field + record-level entries)
  record_reviewer.py     # Post-hoc per-record review pass
  benchmark.py           # Accuracy + latency benchmark (synthetic PHI test set)
  test_data/             # Cached synthetic test fixtures (checked in)
    names_and_dates.json
  prompts/
    generic_phi.jinja
    date_scan.jinja
    name_scan.jinja
    record_review.jinja

scripts/text_phi/operations/
  llm.py                 # LlmScanOperation, LlmDateScanOperation, LlmNameScanOperation

scripts/text_phi/cli.py  # + --llm-config, --llm-server-url, --llm-model, --review-out,
                         #   --auto-apply-llm, --enable-record-review, --llm-cache-clear
```

## Schema shape

```json
{
  "name": "llm_scan",
  "params": {
    "prompt": "generic_phi",
    "context": {
      "patient_name":    "{{ record.subject_name }}",
      "admission_date":  "{{ record.implant_date }}",
      "field_purpose":   "Freeform clinical note from a testing-day report"
    },
    "model_hint": "phi_detector",
    "report_only": true
  }
}
```

Convenience wrappers (`llm_date_scan`, `llm_name_scan`) hard-wire their own prompt name but accept the same `context`, `model_hint`, `report_only` params.

**Context resolution:** each `context.<key>` value is a Jinja template rendered against the current record + record_context. Static strings pass through unchanged.

## Guided response schema (all servers)

```json
{
  "type": "object",
  "properties": {
    "spans": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type":         {"type": "string", "enum": ["DATE","PERSON","PHONE","EMAIL","ADDRESS","MRN","LOCATION","AGE","OTHER_PHI"]},
          "matched_text": {"type": "string"},
          "reason":       {"type": "string"}
        },
        "required": ["type","matched_text","reason"]
      }
    }
  },
  "required": ["spans"]
}
```

Character offsets are NOT requested — LLMs are unreliable at offsets. We ask for `matched_text` + `reason` and resolve offsets client-side by searching for `matched_text` in the value. Ambiguous matches: take first occurrence; flag ambiguity in the report.

Client translates the schema into the right server-specific body:
- **Ollama**: `{"format": <schema>}` in the request body.
- **vLLM**: `{"extra_body": {"guided_json": <schema>}}`.
- Others: fall back to prompt-injected JSON instructions.

## Reproducibility

- `seed` from config (default 42) sent on every request.
- `temperature: 0.0` (also from config).
- Cache key includes `seed` — changing seed re-runs; same seed re-uses cache.
- Ollama and vLLM both honor `seed` in the OpenAI-compatible payload. LM Studio partially.
- Prompts are versioned by filename + content-hash included in the cache key so prompt edits invalidate cache.

## Human-reviewable report

Every LLM finding produces a report entry:

```json
{
  "record_location": {"row": 3},
  "field": "note_text",
  "operation": "llm_scan",
  "value_seen": "The patient's daughter Ann helped translate on 3/15...",
  "recommended_redacted": "The patient's daughter [PERSON] helped translate on [DATE]...",
  "spans": [
    {"start": 22, "end": 25, "type": "PERSON", "matched_text": "Ann",
     "reason": "First name of patient's daughter — indirect identifier"},
    {"start": 38, "end": 42, "type": "DATE",   "matched_text": "3/15",
     "reason": "Specific date not shifted; leaks admission timing"}
  ]
}
```

Two modes per operation (set via `params.report_only`):

- **`report_only: true` (default)** — value passes through unchanged; report captures the recommendation.
- **`report_only: false`** — value is redacted per LLM spans AND the report is written for after-the-fact spot-checking.

Global override via CLI `--auto-apply-llm` (off by default).

## Softer per-record review pass

`RecordReviewer` runs *after* `RecordRedactor.process_records()`. For each record:

1. Concatenate all redacted string fields with delimiters (`---BEGIN <field>---\n<value>\n---END <field>---`).
2. Send to the LLM with `record_review.jinja`.
3. Parse spans → map back to field by delimiter position.
4. Append to `llm_review.json` under `record_review_flags`.

Never auto-modifies — 100% audit.

## Caching

SQLite table:

```
CREATE TABLE llm_cache(
  cache_key   TEXT PRIMARY KEY,
  response    TEXT NOT NULL,
  created_at  INTEGER NOT NULL
)
```

Key = `sha256(f"{server_type}|{model}|{prompt_hash}|{input_hash}|{context_hash}|{seed}|{temperature}")`. Survives across runs. `--llm-cache-clear` purges.

## Benchmark suite (`benchmark.py`)

Synthetic test set generated deterministically (seed-controlled) and cached under `llm/test_data/names_and_dates.json`. Composition:

- **25 name-containing passages** (2–4 sentences each). Names drawn from a diverse pool; embedded in realistic clinical prose.
- **25 date-containing passages** (2–4 sentences each). Dates in mixed formats: `"3/15/2019"`, `"March 15"`, `"three weeks after implant"`, `"the following Tuesday"`, `"early spring 2019"`.
- **10 of these overlap** (contain both a name and a freeform date).

Then perturbation set:
- **Typo variants** of each name (Damerau–Levenshtein 1–2): `"Ann"` → `"Ana"`, `"An"`, `"Anne"`.
- **Case variants**: `"john smith"`, `"JOHN SMITH"`.
- **Nickname variants**: `"John"` → `"Johnny"`, `"Jack"`.
- **Date phrasing variants**: `"March"` → `"Mar"`, `"3"`, `"the third month"`.

Each passage is labeled with ground-truth spans (start, end, type). Benchmark measures:

- **Precision / Recall / F1** per entity type (PERSON, DATE, OTHER_PHI).
- **Latency** per call: p50, p95, p99.
- **Cost** in cache hits vs misses.
- **Comparison against Presidio + subject-name baseline** on the same set — quantifies what the LLM adds.

Command:
```
python -m scripts.text_phi.llm.benchmark \
    --llm-config temp/llm_config.json \
    --output temp/llm_benchmark.json \
    [--regenerate-test-data]
```

The report ends with a "recommendation" section: if F1 < 0.80 for either PERSON or DATE, warn that the model isn't reliable enough for auto-apply mode; recommend upping to a bigger model or disabling auto-apply.

## CLI additions

New flags on `text_phi redact`:

- `--llm-config PATH` — enables LLM ops. Required when the schema uses any `llm_*` operation.
- `--llm-server-url URL` — override server URL from config.
- `--llm-model NAME` — override the concrete model for `phi_detector` hint (convenience for testing).
- `--review-out PATH` — where to write `llm_review.json` (default `<output>.llm_review.json`).
- `--auto-apply-llm` — set `report_only=false` globally. Off by default.
- `--enable-record-review` — run the post-scan `RecordReviewer` pass. Off by default.
- `--llm-cache-clear` — wipe cache before running.

Schema-load-time validation: if any field's operations include an `llm_*` op AND `--llm-config` isn't supplied, raise with a helpful error.

## Schema generation integration

`generate_schema.py` gains `--enable-llm`. When on: for each column currently getting `[subject_name_scan, generic_phi_scan]`, append `llm_scan` with `report_only: true`, `prompt: "generic_phi"`, and `context.patient_name` / `context.admission_date` inherited from `depends_on`. Identifier-flagged and file-type columns are unchanged.

## Tests

Coverage target ≥ 90% for the new `llm/` module. All LLM-dependent tests use a **mocked** HTTP client — no live model required for CI. The benchmark is a separate opt-in script that DOES require a live model.

- `test_client.py` — payload format per server type (Ollama vs vLLM structured-output), retries, timeout, non-200.
- `test_cache.py` — hit/miss, key stability across process restarts, `--llm-cache-clear`, seed propagation into key.
- `test_template.py` — Jinja rendering; per-key context templates; missing-var behavior.
- `test_response.py` — spans → RedactionSpan; matched_text lookup; ambiguity flagging; malformed JSON.
- `test_review_report.py` — report file structure, append semantics.
- `test_llm_operations.py` — `LlmScanOperation` with mock client covering report_only=true (value unchanged), report_only=false (value redacted), context template resolution, empty value passthrough, cache-hit path.
- `test_record_reviewer.py` — concatenation format, per-field attribution from delimiter positions.
- `test_benchmark.py` — synthetic test-set generation is deterministic under a seed; scoring functions produce expected metrics on hand-crafted mock LLM responses.

Existing 89+ inspect tests, 449 pipeline tests still pass.

## Deployment docs

`scripts/text_phi/llm/README.md`:

- **Laptop path (Ollama)**: `brew install ollama`; `ollama serve`; `ollama pull qwen2.5:7b-instruct`. That's it. Copy the sample `llm_config.json` and run.
- **Cluster path (vLLM)**: `pip install vllm`; `vllm serve <model> --tensor-parallel-size 4 ...`; swap config `server_type` + `server_url`.
- Config file field reference.
- Cache management + expected sizes.
- Model-swap workflow (`ollama pull <tag>`, edit config, re-run).
- Benchmark workflow.

## Dependencies

`pyproject.toml` gets `httpx` and `jinja2` in optional-dependencies (`llm` extra). Ollama and vLLM are external installs — not project deps.

## Migration path

Each phase runs `pytest tests/text_phi/` before moving on.

**Phase A — General-purpose LLM client (no PHI logic).** `config.py`, `client.py` (Ollama + vLLM structured-output support), `cache.py`, `template.py`, `response.py`. Tests fully mocked.

**Phase B — PHI operations.** `operations/llm.py` (`LlmScanOperation` + wrappers). Register. Schema layer accepts them.

**Phase C — Report writer.** `review_report.py`. Standalone.

**Phase D — CLI integration.** `cli.py` new flags, schema-validate-time check, wire report writer into redact flow.

**Phase E — Record reviewer.** `record_reviewer.py` + prompt. `--enable-record-review` flag.

**Phase F — Schema generation.** `--enable-llm` in `generate_schema.py`.

**Phase G — Prompts + README.**

**Phase H — Benchmark suite.** Synthetic data generator + scoring + comparison-with-Presidio report.

## End-to-end verification

1. `pytest tests/text_phi/ --cov=scripts/text_phi` — coverage ≥ 90% for `llm/`, existing suite still green.
2. User installs Ollama on their MacBook, pulls `qwen2.5:7b-instruct`, starts `ollama serve`.
3. User runs `python -m scripts.text_phi.llm.benchmark --llm-config temp/llm_config.json --output temp/bench.json`. Reads the F1/latency numbers; decides whether to enable `--auto-apply-llm` or stay report-only.
4. User regenerates schema with `--enable-llm`.
5. User runs `text_phi redact` with `--llm-config`; observes:
   - `temp/llm_cache.sqlite` growing across runs.
   - `temp/data_redacted.csv` where LLM ops have `report_only: true` shows only hard-check redactions.
   - `temp/llm_review.json` populated with per-field findings + record_review flags.
6. User spot-checks the review report; approves/rejects findings. Application script (`apply_review.py`) deferred to Phase 2.
7. Cluster path: user swaps config `server_type: "vllm"` + points at their vLLM endpoint. Zero code changes.

## Out of scope

- **Application script** for approved review findings — deferred.
- **Async client + client-side batching** — add if throughput requires.
- **Per-op model overrides** at schema level.
- **LLM ops on non-string dtypes**.
- **Fine-tuning**, prompt A/B testing.
- **Field-level attribution refinement** for record review beyond delimiter parsing.
