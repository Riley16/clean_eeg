"""Tests for scripts/text_phi/llm/benchmark.py — scoring + generation only.

The live LLM run and Presidio baseline are exercised manually via the
main() CLI. Here we test the deterministic bits: test-set generation,
scoring, and metric aggregation.
"""

from __future__ import annotations

from scripts.text_phi.llm.benchmark import (
    Ground,
    Metrics,
    Passage,
    generate_test_set,
    make_recommendation,
    score_passage,
    summarize,
)


# ---------- generation ----------

def test_generate_test_set_deterministic():
    a = generate_test_set(seed=42)
    b = generate_test_set(seed=42)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert pa.id == pb.id
        assert pa.text == pb.text


def test_generate_covers_25_names_25_dates_10_both():
    passages = generate_test_set(seed=42)
    base = [p for p in passages if p.parent_id is None]
    n_name_only = sum(1 for p in base if p.categories == ("name",))
    n_date_only = sum(1 for p in base if p.categories == ("date",))
    n_both = sum(1 for p in base if set(p.categories) == {"name", "date"})
    assert n_name_only == 25
    assert n_date_only == 25
    assert n_both == 10


def test_generate_includes_perturbations():
    passages = generate_test_set(seed=42)
    perturbed = [p for p in passages if p.parent_id is not None]
    assert perturbed, "expected some perturbations"
    assert any(p.perturbation == "lowercase_name" for p in perturbed)
    assert any(p.perturbation == "damerau_typo" for p in perturbed)


def test_perturbed_passages_have_matching_ground_truth():
    """Perturbed ground_truth values appear in the passage text."""
    passages = generate_test_set(seed=42)
    for p in passages:
        for g in p.ground_truth:
            assert g.matched_text in p.text, (
                f"passage {p.id}: ground_truth {g.matched_text!r} not found "
                f"in text {p.text!r}"
            )


# ---------- Metrics ----------

def test_metrics_precision_recall_f1():
    m = Metrics(tp=8, fp=2, fn=2)
    assert m.precision == 0.8
    assert m.recall == 0.8
    assert abs(m.f1 - 0.8) < 1e-9


def test_metrics_zero_divisor_returns_zero():
    m = Metrics(tp=0, fp=0, fn=0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


# ---------- score_passage ----------

def _passage(text: str, gts: list[tuple[str, str]]) -> Passage:
    return Passage(
        id="x", categories=("name",),
        text=text, ground_truth=[Ground(t, m) for t, m in gts],
    )


def test_score_perfect_match():
    p = _passage("Dr. John Smith was here.", [("PERSON", "John Smith")])
    m = score_passage(p, [("PERSON", "John Smith")])
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)


def test_score_case_insensitive_match():
    p = _passage("hi john smith", [("PERSON", "John Smith")])
    m = score_passage(p, [("PERSON", "john smith")])
    assert m.tp == 1


def test_score_false_positive():
    p = _passage("Dr. John Smith", [("PERSON", "John Smith")])
    m = score_passage(p, [
        ("PERSON", "John Smith"),
        ("PERSON", "Nurse Ann"),  # not in ground truth
    ])
    assert m.tp == 1
    assert m.fp == 1
    assert m.fn == 0


def test_score_false_negative():
    p = _passage("John and Ann were here", [
        ("PERSON", "John"), ("PERSON", "Ann"),
    ])
    m = score_passage(p, [("PERSON", "John")])
    assert m.tp == 1
    assert m.fp == 0
    assert m.fn == 1


def test_score_per_type_bucketing():
    p = Passage(id="x", categories=("name", "date"),
                text="Dr. John saw pt on 3/15",
                ground_truth=[Ground("PERSON", "John"),
                              Ground("DATE", "3/15")])
    per_type: dict = {}
    score_passage(p, [("PERSON", "John"), ("DATE", "3/15")], per_type)
    assert per_type["PERSON"].tp == 1
    assert per_type["DATE"].tp == 1


def test_score_ground_truth_matched_once_only():
    """Two predictions of the same ground-truth phrase — only the first
    should be a TP; the second is a FP."""
    p = _passage("John John", [("PERSON", "John")])
    m = score_passage(p, [("PERSON", "John"), ("PERSON", "John")])
    assert m.tp == 1
    assert m.fp == 1


# ---------- summarize + recommendation ----------

def test_summarize_aggregates():
    p1 = _passage("t1", [("PERSON", "A")])
    p2 = _passage("t2", [("PERSON", "B")])
    from scripts.text_phi.llm.benchmark import RunResult
    per_type: dict = {}
    m1 = score_passage(p1, [("PERSON", "A")], per_type)
    m2 = score_passage(p2, [], per_type)
    summary = summarize("test", [(p1, RunResult(0, []), m1),
                                 (p2, RunResult(0, []), m2)], per_type)
    assert summary["n_passages"] == 2
    assert summary["total"]["tp"] == 1
    assert summary["total"]["fn"] == 1


def test_recommendation_warns_below_threshold():
    llm = {
        "per_type": {
            "PERSON": {"f1": 0.60},
            "DATE": {"f1": 0.85},
        }
    }
    r = make_recommendation(llm)
    assert "do NOT" in r


def test_recommendation_endorses_above_threshold():
    llm = {
        "per_type": {
            "PERSON": {"f1": 0.90},
            "DATE": {"f1": 0.88},
        }
    }
    r = make_recommendation(llm)
    assert "defensible" in r
