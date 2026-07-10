# LLM-driven PHI redaction

This subdirectory ships a local, model-agnostic LLM layer that scans free-text
fields for PHI the hard-check pipeline misses (freeform dates like
`"three weeks after implant"`, family-member names, indirect identifiers,
etc.). Findings are **report-only by default** — the LLM writes a
human-reviewable JSON of proposed spans and the CSV is untouched until you
opt into auto-apply.

## Two deployment paths

### Laptop (default) — Ollama

Local, macOS/Metal + Linux/CUDA, easy model swap.

```bash
# Install
brew install ollama
ollama serve                     # background; listens on :11434

# Pull a model (any Ollama tag works; qwen2.5:7b-instruct is a strong default)
ollama pull qwen2.5:7b-instruct
```

### Cluster — vLLM

Tensor-parallel across multiple GPUs, higher throughput.

```bash
pip install vllm
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len 8192 --port 8000 \
    --gpu-memory-utilization 0.9
```

The client is identical for both — you just change `server_type` and
`server_url` in the config.

## Config file

Copy the template below to `temp/llm_config.json` and edit:

```json
{
  "server_type": "ollama",
  "server_url": "http://localhost:11434/v1",
  "models": {
    "phi_detector":    "qwen2.5:7b-instruct",
    "record_reviewer": "qwen2.5:7b-instruct"
  },
  "cache_path":       "temp/llm_cache.sqlite",
  "seed":             42,
  "temperature":      0.0,
  "timeout_seconds":  60,
  "max_retries":      3,
  "api_key":          null
}
```

### Fields

| Field | Meaning |
|---|---|
| `server_type` | `ollama` / `vllm` / `openai` / `lm_studio` — controls how the structured-output request is formed |
| `server_url` | Base URL that hosts `/chat/completions` |
| `models.phi_detector` | Concrete model tag for per-field scanning |
| `models.record_reviewer` | Concrete model tag for the post-scan record-level review |
| `cache_path` | SQLite cache of prior responses (safe to delete anytime) |
| `seed` | Sent on every request for reproducibility |
| `temperature` | Sent on every request (default 0 = deterministic) |
| `timeout_seconds` | Per-request timeout |
| `max_retries` | Retries on 429/5xx/timeout with exponential backoff |
| `api_key` | Bearer token if the server needs one; usually `null` locally |

## Recommended models for laptop

Runs comfortably on a MacBook with ≥16 GB unified memory (Ollama q4 quant):

| Model | Ollama tag | ~RAM | Notes |
|---|---|---:|---|
| **Qwen 2.5 7B Instruct** | `qwen2.5:7b-instruct` | 5 GB | Recommended default; strong JSON output |
| Llama 3.2 3B Instruct | `llama3.2:3b` | 2.5 GB | Fastest; may miss subtle PHI |
| Qwen 2.5 14B Instruct | `qwen2.5:14b` | 10 GB | Higher accuracy if you have 32 GB |
| Phi-4 14B | `phi4:14b` | 9 GB | Alternative to Qwen 14B |
| Gemma 3 4B | `gemma3:4b` | 3 GB | Middle ground |

## Running

Generate a schema with `llm_scan` operations appended:

```bash
python -m scripts.text_phi.inspect.generate_schema \
    --inspection temp/inspection.json \
    --output temp/redcap.schema.json \
    --enable-llm
```

Redact with the LLM layer active:

```bash
python -m scripts.text_phi.cli redact \
    --input YOUR_DATA.csv \
    --output temp/data_redacted.csv \
    --schema temp/redcap.schema.json \
    --llm-config temp/llm_config.json \
    --audit-out temp/audit.json
```

Optional flags:

- `--llm-server-url URL` — override server URL from the config.
- `--llm-model NAME` — override the concrete model for the `phi_detector` hint.
- `--auto-apply-llm` — apply LLM recommendations to the redacted CSV
  instead of only reporting them.
- `--enable-record-review` — after per-field scanning, run one LLM pass per
  record over all its concatenated fields, looking for anything the
  per-field pass missed. Audit-only; never modifies the CSV.
- `--review-out PATH` — where to write the LLM review report. Default:
  `<output>.llm_review.json`.
- `--llm-cache-clear` — wipe the response cache before starting (useful
  after a prompt edit).

## What the review report looks like

```json
{
  "source": "YOUR_DATA.csv",
  "schema_sha256": "…",
  "n_findings": 12,
  "n_record_flags": 3,
  "findings": [
    {
      "record_location": {"row": 3},
      "field": "note_text",
      "operation": "llm_scan",
      "model": "qwen2.5:7b-instruct",
      "value_seen": "The patient's daughter Ann helped translate on 3/15…",
      "recommended_redacted": "The patient's daughter [PERSON] helped translate on [DATE]…",
      "spans": [
        {"start": 22, "end": 25, "type": "PERSON", "matched_text": "Ann",
         "reason": "First name of patient's daughter — indirect identifier",
         "ambiguous": false}
      ]
    }
  ],
  "record_flags": [ … same shape, entries produced by --enable-record-review … ]
}
```

Human review workflow: open the JSON, filter to findings you want to
accept, and (once we ship the follow-up `apply_review.py`) apply the
accepted subset to the CSV. Until then, `--auto-apply-llm` applies all
findings blindly — use with caution.

## Cache management

The response cache is a small SQLite file keyed on
`(server_type, model, prompt_hash, input_hash, context_hash, seed, temperature)`.
Rerunning on the same input with the same seed skips the LLM entirely.

- Size: typically a few MB for a REDCap-scale project.
- Clear: pass `--llm-cache-clear` or delete the file.
- Any change to prompts, seed, model, or input invalidates that entry.

## Prompt library

The four Jinja prompts live in `prompts/`:

| Prompt | Used by |
|---|---|
| `generic_phi.jinja` | `llm_scan` — general PHI detection |
| `date_scan.jinja`   | `llm_date_scan` — dates only |
| `name_scan.jinja`   | `llm_name_scan` — names only |
| `record_review.jinja` | `RecordReviewer` (via `--enable-record-review`) |

Edit them freely; changes take effect on the next run (the cache key
includes the prompt hash, so old responses are invalidated automatically).

## Benchmarking (deferred to Phase H)

Coming soon: `python -m scripts.text_phi.llm.benchmark …` runs a synthetic
name-and-date test set through the LLM and Presidio, reports F1 and
latency per entity type, and warns if the model is too weak for
`--auto-apply-llm`.
