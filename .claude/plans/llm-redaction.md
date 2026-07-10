# Plan: LLM agent redaction pathway — status + runbook

## Where we are (2026-07-09)

**Architecture**: shipped. Approved plan v2 (local-first, model-agnostic, human-review-first) is fully implemented across 8 phases. All 603 unit tests pass on `text-phi-redaction` branch. The design decisions locked with the user (Ollama laptop / vLLM cluster interchangeable via config, model-agnostic through `model_hint` indirection, seed + temperature=0 for reproducibility, `report_only: true` default per LLM op, benchmark with Presidio comparison and F1 threshold) are all in code.

Relevant commits (branch: `text-phi-redaction`):

- `764cf1c` `feat(text_phi): LLM redaction pathway — local-first, model-agnostic` (LLM subpackage, prompts, benchmark, README)
- `a45f7c5` `feat(text_phi): wire LLM path into records, CLI, and schema generation` (Records / CLI flags / --enable-llm)
- `ba9fc90` `docs: text-PHI and LLM redaction design plans`
- `32a85bc` `feat(text_phi): REDCap CSV inspection + schema derivation tools`
- `fd048cf` `feat(text_phi): schema-driven PHI redaction pipeline for .txt and CSV`

**Runtime**: not yet stood up. Ollama isn't installed on the laptop, no `temp/llm_config.json`, no cache DB. Before the benchmark can execute, the setup section below has to run.

**What's left**: (1) local Ollama install + first model pull, (2) benchmark protocol executed with high-level user guidance, (3) decision on whether the initial model is trustworthy enough for `--auto-apply-llm` or should stay report-only.

## What's built

Source under `scripts/text_phi/llm/`:

| File | Purpose |
|---|---|
| `config.py` | `LLMConfig` + `ServerType` enum + model registry |
| `client.py` | Sync httpx OpenAI-compatible client; per-server-type structured output |
| `cache.py` | SQLite content-hash cache keyed on `(server, model, prompt_hash, input_hash, context_hash, seed, temp)` |
| `template.py` | Jinja rendering + `PromptRegistry` |
| `response.py` | PHI span JSON schema + `matched_text`→offset resolution |
| `review_report.py` | Human-reviewable JSON writer (findings + record-flags buckets) |
| `record_reviewer.py` | Post-scan per-record LLM audit pass |
| `benchmark.py` | Accuracy + latency benchmark; Presidio baseline; F1 recommendation |
| `prompts/{generic_phi,date_scan,name_scan,record_review}.jinja` | Editable prompt templates |
| `test_data/names_and_dates.json` | 141-passage synthetic benchmark fixture (checked in) |
| `README.md` | Deployment + config + model docs |

Operations at `scripts/text_phi/operations/llm.py`: `llm_scan`, `llm_date_scan`, `llm_name_scan` (all registered in the operations registry, all `string` dtype only, all default `report_only: true`).

Wired in:

- `records.py` — `OperationContext` + `RecordRedactor` carry optional `llm_client` / `llm_cache` / `prompt_registry` / `review_report` / `record_location`.
- `cli.py` `redact` subcommand — new flags:
  - `--llm-config PATH` (required if schema has any `llm_*` op)
  - `--llm-server-url URL`, `--llm-model NAME` (overrides)
  - `--review-out PATH` (default `<output>.llm_review.json`)
  - `--auto-apply-llm` (off by default — LLM findings report-only)
  - `--enable-record-review` (post-scan LLM audit)
  - `--llm-cache-clear`
- `inspect/generate_schema.py` — `--enable-llm` appends `llm_scan` with `report_only: true` to every field currently getting `[subject_name_scan, generic_phi_scan]`.

Tests: 118 new mocked LLM tests, all using `httpx.MockTransport`. No live model required for CI.

## LLM setup steps (do these before the benchmark)

### 1. Install Ollama

```bash
brew install ollama
```

### 2. Start the server

Leave this running in a background terminal (or use `brew services start ollama`):

```bash
ollama serve
```

Confirm it responded on port 11434:

```bash
curl -s http://localhost:11434/api/version
```

### 3. Pull a model

Default recommendation for a MacBook with ≥16 GB unified memory:

```bash
ollama pull qwen2.5:7b-instruct
```

Alternatives to consider (in order of speed → accuracy):

| Tag | Size | RAM | Use case |
|---|---|---:|---|
| `llama3.2:3b` | 3B | ~2.5 GB | Fastest, iteration |
| `qwen2.5:7b-instruct` | 7B | ~5 GB | Recommended default |
| `gemma3:4b` | 4B | ~3 GB | Alternative small |
| `qwen2.5:14b` | 14B | ~10 GB | Higher accuracy (if 32 GB) |
| `phi4:14b` | 14B | ~9 GB | Alternative mid-size |

### 4. Write the config file

Save as `temp/llm_config.json` (the redact CLI + benchmark both read this):

```json
{
  "server_type": "ollama",
  "server_url": "http://localhost:11434/v1",
  "models": {
    "phi_detector":    "qwen2.5:7b-instruct",
    "record_reviewer": "qwen2.5:7b-instruct"
  },
  "cache_path":      "temp/llm_cache.sqlite",
  "seed":            42,
  "temperature":     0.0,
  "timeout_seconds": 60,
  "max_retries":     3
}
```

### 5. Sanity check

Confirm the client can round-trip a request:

```bash
python -c "
from scripts.text_phi.llm.config import LLMConfig
from scripts.text_phi.llm.client import LLMClient
cfg = LLMConfig.load('temp/llm_config.json')
with LLMClient(cfg) as c:
    r = c.chat([{'role':'user','content':'Reply with the JSON: {\"ok\": true}'}],
               response_schema={'type':'object','properties':{'ok':{'type':'boolean'}},
                                'required':['ok']})
    print(r['choices'][0]['message']['content'])
"
```

Expect `{"ok": true}` (or a valid JSON with an `ok` field).

## Performance-testing protocol

The plan is: I run the benchmark autonomously against a small set of candidate models, report the numbers, and surface the go/no-go decision to the user. The user's inputs are limited to (a) which model(s) to try beyond the default and (b) whether to accept the winning model for `--auto-apply-llm`.

### What I'll do without asking

1. **Confirm setup** — `curl http://localhost:11434/api/version` succeeds and the sanity-check snippet returns valid JSON.

2. **Baseline benchmark** — Qwen 2.5 7B Instruct:

   ```bash
   python -m scripts.text_phi.llm.benchmark \
       --llm-config temp/llm_config.json \
       --output temp/bench_qwen25_7b.json
   ```

   The synthetic set is 141 passages. Expect ~10–30 min on a MacBook depending on model speed.

3. **Comparison models** — repeat with two other models the user picks below, editing `models.phi_detector` in the config between runs (and pointing `--output` at a distinct file per model):

   ```bash
   ollama pull <model>
   # edit temp/llm_config.json → models.phi_detector: <model>
   python -m scripts.text_phi.llm.benchmark --llm-config temp/llm_config.json --output temp/bench_<model>.json
   ```

4. **Summary report** — I'll pull the numbers from each JSON into a short table with:
   - PERSON precision / recall / F1
   - DATE precision / recall / F1
   - Latency p50 / p95
   - Cache hit rate
   - Presidio-baseline delta (LLM F1 minus Presidio F1 per type)
   - Recommendation string from each report

5. **Cost audit** — I'll `du -h temp/llm_cache.sqlite` after the runs to confirm cache size is reasonable and report the disk footprint.

### What I need from the user (before I start)

1. Which 1–2 alternative models to benchmark alongside Qwen 2.5 7B (from the table above, or user-supplied).
2. F1 acceptance threshold for `--auto-apply-llm` — plan default is 0.80 per entity type, ask if the user wants to raise it (e.g. 0.90) for a clinical dataset.
3. Confirmation that the initial-load Ollama model download (~5 GB per model) is fine to run over their network.

### What I will NOT do without asking

- Enable `--auto-apply-llm` on real data. Even after benchmarking, the CSV pipeline stays in report-only mode until the user explicitly opts in.
- Run the benchmark against the real REDCap CSV. The 141-passage synthetic set is the acceptance test.
- Modify the prompts. If benchmark numbers are low, I'll propose prompt edits and ask before applying.
- Push the branch to any remote or open a PR.
- Delete the cache DB. Every rerun is cheap because of the SHA-256 keyed cache.

### Go / no-go decision matrix

After the benchmark, per model tested:

| PERSON F1 | DATE F1 | My recommendation |
|---|---|---|
| ≥ 0.90 | ≥ 0.90 | Safe for `--auto-apply-llm` on this dataset. |
| ≥ 0.80 | ≥ 0.80 | OK for `--auto-apply-llm` **if** paired with human spot-check of the review report. |
| Either < 0.80 | Either < 0.80 | Stay report-only. Try a larger model, then decide. |

The `benchmark.py` output already contains a `"recommendation"` string encoding this — I'll surface it verbatim in my summary.

## Runbook after benchmarks pass

Once we've picked a model:

1. Update `temp/llm_config.json` to lock the winner in `models.phi_detector` and `models.record_reviewer`.

2. Regenerate the schema with LLM ops appended:

   ```bash
   python -m scripts.text_phi.inspect.generate_schema \
       --inspection temp/inspection.json \
       --output temp/redcap.schema.json \
       --enable-llm
   ```

3. Redact:

   ```bash
   python -m scripts.text_phi.cli redact \
       --input YOUR_DATA.csv \
       --output temp/data_redacted.csv \
       --schema temp/redcap.schema.json \
       --llm-config temp/llm_config.json \
       --audit-out temp/audit.json
   ```

   `temp/data_redacted.csv.llm_review.json` collects the LLM's proposed spans. The redacted CSV is hard-check-only until the user adds `--auto-apply-llm`.

4. Optional post-scan record-level audit (adds ~3975 more LLM calls, one per record):

   ```
   ... redact command as above ... --enable-record-review
   ```

## Deferred (Phase 2, not on this iteration)

- `apply_review.py` — script that reads the `llm_review.json`, lets the user accept/reject findings, and applies the accepted subset to the CSV.
- Async client + client-side batching if the sequential-per-field throughput becomes a bottleneck at REDCap scale.
- Per-op model overrides at schema level (currently one model per `model_hint` in the config).
- LLM ops on non-string dtypes.
- Prompt A/B testing / versioning.

## Out of scope for this plan

- Fine-tuning any model.
- vLLM cluster setup (documented in the module README; not on the critical path for the REDCap CSV).
- Streaming responses (not needed given per-field small requests).
