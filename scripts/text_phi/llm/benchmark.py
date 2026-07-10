"""Accuracy + latency benchmark for the LLM redaction path.

Generates a deterministic synthetic test set of 2–4 sentence passages with
embedded names and dates, plus perturbations (typos, case flips, nickname
substitutions, date-phrasing variants), then measures precision / recall /
F1 / latency of the LLM against a Presidio + subject-name baseline.

The synthetic test set is seed-controlled and cached under
`scripts/text_phi/llm/test_data/names_and_dates.json` — pass
`--regenerate-test-data` to rebuild.

Usage:
    python -m scripts.text_phi.llm.benchmark \\
        --llm-config temp/llm_config.json \\
        --output temp/llm_benchmark.json \\
        [--regenerate-test-data]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .cache import LLMCache, build_cache_key, hash_str
from .client import LLMClient
from .config import LLMConfig
from .response import (
    LlmResponseError,
    PHI_SPAN_SCHEMA,
    parse_response,
    resolve_spans,
)
from .template import PromptRegistry, render_string


TEST_DATA_PATH = Path(__file__).parent / "test_data" / "names_and_dates.json"


# ---------- test-data generation ----------

_NAME_POOL: tuple[tuple[str, str], ...] = (
    ("John", "Smith"), ("Ann", "Johnson"), ("Michael", "Chen"),
    ("Maria", "Rodriguez"), ("David", "Nguyen"), ("Sarah", "Williams"),
    ("James", "Brown"), ("Emily", "Davis"), ("Robert", "Miller"),
    ("Jennifer", "Wilson"), ("Priya", "Patel"), ("Wei", "Zhang"),
    ("Aisha", "Ali"), ("Carlos", "Garcia"), ("Ling", "Wu"),
    ("Yusuf", "Ahmed"), ("Olivia", "Martinez"), ("Ethan", "Anderson"),
    ("Sophia", "Thomas"), ("Liam", "Taylor"), ("Amelia", "Moore"),
    ("Noah", "Jackson"), ("Isabella", "White"), ("Ava", "Harris"),
    ("Mason", "Martin"),
)

_DATE_LITERAL_POOL: tuple[str, ...] = (
    "3/15/2019", "March 15, 2019", "2019-03-15", "Mar 15", "March 15",
    "07-04-2020", "December 25, 2021", "1/1/2022",
    "the following Tuesday", "three weeks after implant",
    "two months post-op", "the day after admission",
    "early spring 2019", "mid-March", "late fall of 2020",
    "Christmas 2020", "the fourth of July",
)

_NAME_TEMPLATES: tuple[str, ...] = (
    "The patient was seen by Dr. {name} for the initial workup. "
    "Vitals were stable. Follow-up scheduled.",
    "{name} performed the amplitude determination. No afterdischarges "
    "were noted. The patient tolerated the session well.",
    "Family present at bedside: {name}. The patient's daughter helped "
    "translate the discussion.",
    "Provider handoff to {name} at shift change. Recording continued "
    "without interruption.",
    "Nurse {name} escalated concern about the patient's fatigue. "
    "Testing paused briefly.",
)

_DATE_TEMPLATES: tuple[str, ...] = (
    "The patient was admitted on {date}. Testing began the next day "
    "and continued through the week.",
    "First seizure recorded on {date}. No further events over 48 hours "
    "of monitoring.",
    "Wada testing performed {date}. Results supported left-hemisphere "
    "language dominance.",
    "Explant scheduled for {date} pending pathology review. Family has "
    "been notified.",
    "Post-op MRI on {date} showed no new lesions. Wound healing well.",
)

_BOTH_TEMPLATES: tuple[str, ...] = (
    "{name} was seen on {date} for the annual follow-up. All AEDs at "
    "their previous doses; no side effects reported.",
    "Note by {name} on {date}: patient reports mild HA, no new seizures. "
    "Continuing current regimen.",
    "Recording session with {name} on {date} completed without issue. "
    "Task 4 was skipped due to fatigue.",
)


@dataclass
class Ground:
    type: str  # "PERSON" or "DATE"
    matched_text: str


@dataclass
class Passage:
    id: str
    categories: tuple[str, ...]
    text: str
    ground_truth: list[Ground] = field(default_factory=list)
    parent_id: str | None = None
    perturbation: str | None = None


def _fill_name_passage(rng: random.Random, i: int) -> Passage:
    tmpl = _NAME_TEMPLATES[i % len(_NAME_TEMPLATES)]
    first, last = _NAME_POOL[i]
    full = f"{first} {last}"
    text = tmpl.format(name=full)
    return Passage(
        id=f"name_{i+1:02d}", categories=("name",),
        text=text, ground_truth=[Ground("PERSON", full)],
    )


def _fill_date_passage(rng: random.Random, i: int) -> Passage:
    tmpl = _DATE_TEMPLATES[i % len(_DATE_TEMPLATES)]
    date_str = _DATE_LITERAL_POOL[i % len(_DATE_LITERAL_POOL)]
    text = tmpl.format(date=date_str)
    return Passage(
        id=f"date_{i+1:02d}", categories=("date",),
        text=text, ground_truth=[Ground("DATE", date_str)],
    )


def _fill_both_passage(rng: random.Random, i: int) -> Passage:
    tmpl = _BOTH_TEMPLATES[i % len(_BOTH_TEMPLATES)]
    first, last = _NAME_POOL[(i * 3 + 5) % len(_NAME_POOL)]
    full = f"{first} {last}"
    date_str = _DATE_LITERAL_POOL[(i * 5 + 2) % len(_DATE_LITERAL_POOL)]
    text = tmpl.format(name=full, date=date_str)
    return Passage(
        id=f"both_{i+1:02d}", categories=("name", "date"),
        text=text,
        ground_truth=[Ground("PERSON", full), Ground("DATE", date_str)],
    )


def generate_test_set(seed: int = 42) -> list[Passage]:
    """Deterministic synthetic passage set.

    Composition: 25 name-only + 25 date-only + 10 overlap (name + date).
    Then perturbations of each base passage.
    """
    rng = random.Random(seed)
    passages: list[Passage] = []
    for i in range(25):
        passages.append(_fill_name_passage(rng, i))
    for i in range(25):
        passages.append(_fill_date_passage(rng, i))
    for i in range(10):
        passages.append(_fill_both_passage(rng, i))
    # Add perturbations for each base passage.
    perturbed: list[Passage] = []
    for base in passages:
        perturbed.extend(_perturb(base, rng))
    return passages + perturbed


def _perturb(base: Passage, rng: random.Random) -> list[Passage]:
    """Generate a small set of perturbations for a base passage."""
    out: list[Passage] = []
    # Case perturbation applies to name passages.
    if "name" in base.categories:
        # Lowercase the entire name (not the whole passage — that would
        # change ground_truth too much).
        for g in base.ground_truth:
            if g.type != "PERSON":
                continue
            new_name_lc = g.matched_text.lower()
            new_text = base.text.replace(g.matched_text, new_name_lc, 1)
            new_gt = [
                Ground(gg.type, new_name_lc if gg is g else gg.matched_text)
                for gg in base.ground_truth
            ]
            out.append(Passage(
                id=f"{base.id}__lower",
                categories=base.categories, text=new_text,
                ground_truth=new_gt,
                parent_id=base.id, perturbation="lowercase_name",
            ))
            # Typo: swap two middle characters.
            typo = _damerau_typo(g.matched_text, rng)
            if typo != g.matched_text:
                new_text = base.text.replace(g.matched_text, typo, 1)
                new_gt = [
                    Ground(gg.type, typo if gg is g else gg.matched_text)
                    for gg in base.ground_truth
                ]
                out.append(Passage(
                    id=f"{base.id}__typo",
                    categories=base.categories, text=new_text,
                    ground_truth=new_gt,
                    parent_id=base.id, perturbation="damerau_typo",
                ))
    if "date" in base.categories:
        for g in base.ground_truth:
            if g.type != "DATE":
                continue
            # Numeric → prose swap if the string is a slash format.
            alt = _alt_date_form(g.matched_text)
            if alt:
                new_text = base.text.replace(g.matched_text, alt, 1)
                new_gt = [
                    Ground(gg.type, alt if gg is g else gg.matched_text)
                    for gg in base.ground_truth
                ]
                out.append(Passage(
                    id=f"{base.id}__altdate",
                    categories=base.categories, text=new_text,
                    ground_truth=new_gt,
                    parent_id=base.id, perturbation="alt_date_form",
                ))
    return out


def _damerau_typo(name: str, rng: random.Random) -> str:
    """Swap two adjacent characters or replace one — Damerau-Levenshtein
    distance 1 or 2. Falls back to the original if too short."""
    if len(name) < 4:
        return name
    idx = rng.randint(1, len(name) - 3)
    chars = list(name)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def _alt_date_form(date_str: str) -> str | None:
    """Return a semantically-equivalent alternate rendering of a date, or
    None if we don't have a mapping. Coarse — mainly for demonstration."""
    swaps = {
        "3/15/2019": "March 15, 2019",
        "March 15": "3/15",
        "2019-03-15": "March 15, 2019",
        "07-04-2020": "the fourth of July, 2020",
        "December 25, 2021": "Christmas 2021",
    }
    return swaps.get(date_str)


def _to_json(passages: list[Passage]) -> str:
    return json.dumps([asdict(p) for p in passages], indent=2)


def load_test_set() -> list[Passage]:
    raw = json.loads(TEST_DATA_PATH.read_text(encoding="utf-8"))
    return [
        Passage(
            id=r["id"], categories=tuple(r["categories"]),
            text=r["text"],
            ground_truth=[Ground(**g) for g in r["ground_truth"]],
            parent_id=r.get("parent_id"),
            perturbation=r.get("perturbation"),
        )
        for r in raw
    ]


def save_test_set(passages: list[Passage]) -> None:
    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_DATA_PATH.write_text(_to_json(passages), encoding="utf-8")


# ---------- scoring ----------

@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def score_passage(
    passage: Passage,
    predicted: list[tuple[str, str]],  # (type, matched_text) pairs
    per_type: dict[str, Metrics] | None = None,
) -> Metrics:
    """Match predictions against ground truth. Case-insensitive substring
    match on `matched_text` within-type. Each ground truth can only be
    matched once."""
    metrics = Metrics()
    per_type = per_type if per_type is not None else {}
    unmatched_gt = [(g.type, g.matched_text) for g in passage.ground_truth]

    for p_type, p_text in predicted:
        found = None
        for idx, (g_type, g_text) in enumerate(unmatched_gt):
            if g_type != p_type:
                continue
            if p_text.strip().lower() == g_text.strip().lower():
                found = idx
                break
        if found is not None:
            metrics.tp += 1
            per_type.setdefault(p_type, Metrics()).tp += 1
            unmatched_gt.pop(found)
        else:
            metrics.fp += 1
            per_type.setdefault(p_type, Metrics()).fp += 1

    for g_type, _ in unmatched_gt:
        metrics.fn += 1
        per_type.setdefault(g_type, Metrics()).fn += 1

    return metrics


# ---------- runners ----------

@dataclass
class RunResult:
    latency_ms: float
    predicted_spans: list[tuple[str, str]]


def _run_llm(
    passage: Passage,
    client: LLMClient,
    prompt_registry: PromptRegistry,
    cache: LLMCache | None,
    prompt_name: str = "generic_phi",
) -> RunResult:
    template = prompt_registry.get(prompt_name)
    prompt_text = render_string(template, {"value": passage.text, "context": {}})
    cfg = client.config
    model = cfg.resolve_model("phi_detector")
    cache_key = build_cache_key(
        server_type=cfg.server_type.value, model=model,
        prompt_hash=hash_str(template), input_hash=hash_str(passage.text),
        context_hash=hash_str("{}"), seed=cfg.seed, temperature=cfg.temperature,
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            spans = parse_response(cached)
            resolved = resolve_spans(passage.text, spans)
            return RunResult(0.0, [(s.type, s.matched_text) for s in resolved])

    t0 = time.perf_counter()
    resp = client.chat(
        messages=[{"role": "user", "content": prompt_text}],
        model_hint="phi_detector",
        response_schema=PHI_SPAN_SCHEMA,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LlmResponseError(f"malformed envelope: {e}") from e
    if cache is not None:
        cache.put(cache_key, content)
    spans = parse_response(content)
    resolved = resolve_spans(passage.text, spans)
    return RunResult(dt_ms, [(s.type, s.matched_text) for s in resolved])


def _run_presidio(passage: Passage) -> RunResult:
    """Baseline: Presidio's PERSON + DATE_TIME recognizers only."""
    from ..detectors import build_generic_analyzer
    analyzer = build_generic_analyzer()
    t0 = time.perf_counter()
    results = analyzer.analyze(
        text=passage.text, entities=["PERSON", "DATE_TIME"], language="en",
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    predicted: list[tuple[str, str]] = []
    for r in results:
        span_text = passage.text[r.start:r.end]
        p_type = "PERSON" if r.entity_type == "PERSON" else "DATE"
        predicted.append((p_type, span_text))
    return RunResult(dt_ms, predicted)


# ---------- report ----------

def summarize(
    label: str,
    per_passage: list[tuple[Passage, RunResult, Metrics]],
    per_type: dict[str, Metrics],
) -> dict[str, Any]:
    total = Metrics()
    latencies = [rr.latency_ms for _, rr, _ in per_passage if rr.latency_ms > 0]
    for _, _, m in per_passage:
        total.tp += m.tp
        total.fp += m.fp
        total.fn += m.fn
    summary: dict[str, Any] = {
        "label": label,
        "n_passages": len(per_passage),
        "total": {
            "precision": total.precision,
            "recall": total.recall,
            "f1": total.f1,
            "tp": total.tp, "fp": total.fp, "fn": total.fn,
        },
        "per_type": {
            t: {
                "precision": m.precision, "recall": m.recall, "f1": m.f1,
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
            }
            for t, m in per_type.items()
        },
    }
    if latencies:
        summary["latency_ms"] = {
            "n": len(latencies),
            "p50": statistics.median(latencies),
            "p95": statistics.quantiles(latencies, n=20)[-1]
                if len(latencies) >= 20 else max(latencies),
            "mean": statistics.mean(latencies),
        }
    return summary


def make_recommendation(llm_summary: dict[str, Any]) -> str:
    per_type = llm_summary.get("per_type", {})
    person_f1 = per_type.get("PERSON", {}).get("f1", 0.0)
    date_f1 = per_type.get("DATE", {}).get("f1", 0.0)
    if person_f1 < 0.80 or date_f1 < 0.80:
        return (
            f"PERSON F1={person_f1:.2f}, DATE F1={date_f1:.2f}. "
            "Both under 0.80 → do NOT enable --auto-apply-llm. Prefer a "
            "larger model or accept report-only mode."
        )
    return (
        f"PERSON F1={person_f1:.2f}, DATE F1={date_f1:.2f}. "
        "Both ≥ 0.80 → the model is reasonably reliable. --auto-apply-llm "
        "is defensible if paired with human spot-check of the review report."
    )


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--regenerate-test-data", action="store_true", default=False)
    p.add_argument("--skip-presidio", action="store_true", default=False,
                   help="Skip the Presidio baseline (faster; useful during dev).")
    args = p.parse_args(argv)

    if args.regenerate_test_data or not TEST_DATA_PATH.exists():
        passages = generate_test_set(seed=42)
        save_test_set(passages)
        print(f"Wrote {len(passages)} synthetic passages to {TEST_DATA_PATH}",
              file=sys.stderr)
    passages = load_test_set()

    cfg = LLMConfig.load(args.llm_config)
    client = LLMClient(cfg)
    cache = LLMCache(cfg.cache_path) if cfg.cache_path else None
    prompt_registry = PromptRegistry(Path(__file__).parent / "prompts")

    try:
        llm_records: list[tuple[Passage, RunResult, Metrics]] = []
        llm_per_type: dict[str, Metrics] = {}
        for i, passage in enumerate(passages):
            try:
                rr = _run_llm(passage, client, prompt_registry, cache)
            except LlmResponseError as e:
                print(f"[{i}] LLM error on {passage.id}: {e}", file=sys.stderr)
                rr = RunResult(0.0, [])
            m = score_passage(passage, rr.predicted_spans, llm_per_type)
            llm_records.append((passage, rr, m))

        llm_summary = summarize("llm", llm_records, llm_per_type)

        report: dict[str, Any] = {
            "test_set_size": len(passages),
            "model": cfg.resolve_model("phi_detector"),
            "server_type": cfg.server_type.value,
            "seed": cfg.seed,
            "temperature": cfg.temperature,
            "llm": llm_summary,
        }

        if not args.skip_presidio:
            presidio_records: list[tuple[Passage, RunResult, Metrics]] = []
            presidio_per_type: dict[str, Metrics] = {}
            for passage in passages:
                rr = _run_presidio(passage)
                m = score_passage(passage, rr.predicted_spans, presidio_per_type)
                presidio_records.append((passage, rr, m))
            report["presidio"] = summarize(
                "presidio", presidio_records, presidio_per_type
            )

        report["recommendation"] = make_recommendation(llm_summary)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote benchmark report → {out}", file=sys.stderr)
        print(f"Recommendation: {report['recommendation']}", file=sys.stderr)
    finally:
        if cache is not None:
            cache.close()
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
