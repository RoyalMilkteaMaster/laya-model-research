"""Tests for bootstrap_ci.py on a small synthetic run (no model, no Data Root).

Run:  source multihop_benchmark/scripts/env.sh && uv run --project multihop_benchmark pytest paper/analysis
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("bootstrap_ci.py")
CONDS = ["recall_llm", "rag_llm", "recall_laya", "rag_laya"]
# Correct answers per condition out of 4 questions (q0..q3): EM 25, 50, 50, 100 %.
CORRECT = {"recall_llm": {"q0"}, "rag_llm": {"q0", "q1"}, "recall_laya": {"q0", "q2"}, "rag_laya": {"q0", "q1", "q2", "q3"}}


def make_run(tmp_path, summary_em=None):
    run = tmp_path / "runs" / "main-20260930"
    (run / "records").mkdir(parents=True)
    for c in CONDS:
        lines = []
        for q in ["q0", "q1", "q2", "q3"]:
            ok = q in CORRECT[c]
            lines.append({"dataset": "hotpotqa", "model": "qwen3.5:2b", "condition": c, "question_id": q,
                          "em": 1 if ok else 0, "status": "done"})
        # rag_laya: q3 first written as a timeout, then redone (last line wins).
        if c == "rag_laya":
            lines.insert(0, {"dataset": "hotpotqa", "model": "qwen3.5:2b", "condition": c, "question_id": "q3",
                             "em": 0, "status": "timeout"})
        # recall_llm: q1 is a non-done question that still carries em=1 -> counted as 0.
        if c == "recall_llm":
            lines[1] = dict(lines[1], em=1, status="invalid_tool")
        path = run / "records" / f"hotpotqa__qwen3.5-2b__{c}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    em = {"recall_llm": "25.0", "rag_llm": "50.0", "recall_laya": "50.0", "rag_laya": "100.0"}
    em.update(summary_em or {})
    with open(run / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "condition", "n", "em"])
        for c in CONDS:
            w.writerow(["hotpotqa", "qwen3.5:2b", c, "4", em[c]])
    return tmp_path / "runs"


def run_script(runs, out, b=200, seed=None):
    return subprocess.run([sys.executable, str(SCRIPT), "--runs", str(runs), "--out-dir", str(out / "analysis"),
                           "--table", str(out / "tables" / "bootstrap_ci.tex"), "-B", str(b)]
                          + ([] if seed is None else ["--seed", str(seed)]),
                          capture_output=True, text=True)


def rows(out):
    with open(out / "analysis" / "bootstrap_ci.csv", encoding="utf-8") as f:
        return {r["effect"]: r for r in csv.DictReader(f) if (r["dataset"], r["model"]) == ("hotpotqa", "qwen3.5:2b")}


def test_point_estimates_follow_last_line_and_non_done_rules(tmp_path):
    runs = make_run(tmp_path)
    p = run_script(runs, tmp_path / "out")
    assert p.returncode == 0, p.stdout + p.stderr
    r = rows(tmp_path / "out")
    assert r["controller_rag"]["estimate"] == "50.0"      # 100 - 50
    assert r["controller_recall"]["estimate"] == "25.0"   # 50 - 25
    assert r["retrieval_llm"]["estimate"] == "25.0"       # 50 - 25
    assert r["interaction"]["estimate"] == "25.0"         # 50 - 25
    assert r["interaction"]["n"] == "4" and r["interaction"]["B"] == "200" and r["interaction"]["seed"] == "20261001"
    for e in r.values():
        assert float(e["ci_low"]) <= float(e["estimate"]) <= float(e["ci_high"])


def test_output_is_byte_identical_across_runs(tmp_path):
    runs = make_run(tmp_path)
    assert run_script(runs, tmp_path / "a").returncode == 0
    assert run_script(runs, tmp_path / "b").returncode == 0
    for rel in ("analysis/bootstrap_ci.csv", "tables/bootstrap_ci.tex"):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_mismatch_with_summary_exits_nonzero(tmp_path):
    runs = make_run(tmp_path, summary_em={"rag_laya": "95.0"})
    p = run_script(runs, tmp_path / "out")
    assert p.returncode != 0
    assert "summary.csv" in p.stdout + p.stderr


def test_table_is_bare_tabular_with_row_keys(tmp_path):
    runs = make_run(tmp_path)
    assert run_script(runs, tmp_path / "out").returncode == 0
    tex = (tmp_path / "out" / "tables" / "bootstrap_ci.tex").read_text(encoding="utf-8")
    assert tex.lstrip().startswith("\\begin{tabular}") and tex.rstrip().endswith("\\end{tabular}")
    assert "\\caption" not in tex and "\\label" not in tex and "\\begin{table}" not in tex
    row = [line for line in tex.splitlines() if "% dataset=hotpotqa model=qwen3.5:2b" in line]
    assert len(row) == 1 and "50.0" in row[0]


# --------------------------------------------------------------------------- pairing, percentile, seed
# 40 questions q00..q39 (n chosen so the bootstrap distributions below are Binomial(40, p)):
#   recall_llm  correct on none
#   recall_laya correct on q00..q19
#   rag_llm     correct on q30..q39
#   rag_laya    correct on q00..q19 and q30..q39
# Per question, Laya - LLM is 1 on q00..q19 and 0 elsewhere under both evidence sources, so the
# per-question interaction is 0 everywhere and RAG+Laya - RAG+LLM is Binomial(40, 0.5) / 40.
N40 = [f"q{i:02d}" for i in range(40)]
CORRECT40 = {"recall_llm": set(), "recall_laya": set(N40[:20]), "rag_llm": set(N40[30:]),
             "rag_laya": set(N40[:20]) | set(N40[30:])}


def write_run(tmp_path, correct, qids=None):
    """Synthetic run; summary.csv em is computed here from the correct sets (independent of the script)."""
    run = tmp_path / "runs" / "main-20260930"
    (run / "records").mkdir(parents=True)
    rows = []
    for c in CONDS:
        ids = (qids or {}).get(c, N40)
        lines = [{"dataset": "hotpotqa", "model": "qwen3.5:2b", "condition": c, "question_id": q,
                  "em": int(q in correct[c]), "status": "done"} for q in ids]
        (run / "records" / f"hotpotqa__qwen3.5-2b__{c}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        rows.append(["hotpotqa", "qwen3.5:2b", c, str(len(ids)), f"{100 * len(correct[c] & set(ids)) / len(ids):.1f}"])
    with open(run / "summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["dataset", "model", "condition", "n", "em"]] + rows)
    return tmp_path / "runs"


def test_resampling_is_paired_across_the_four_conditions(tmp_path):
    # The per-question interaction is 0 for every question, so any resample that draws the same
    # questions for all four conditions gives exactly 0; unpaired draws give a nonzero spread.
    assert run_script(write_run(tmp_path, CORRECT40), tmp_path / "out", b=2000).returncode == 0
    r = rows(tmp_path / "out")
    assert (r["interaction"]["estimate"], r["interaction"]["ci_low"], r["interaction"]["ci_high"]) == ("0.0", "0.0", "0.0")


def test_identical_conditions_give_a_zero_width_interval(tmp_path):
    # RAG+Laya answers exactly the questions RAG+LLM answers: the paired difference is 0 in every resample.
    same = dict(CORRECT40, rag_laya=set(CORRECT40["rag_llm"]))
    assert run_script(write_run(tmp_path, same), tmp_path / "out", b=2000).returncode == 0
    r = rows(tmp_path / "out")["controller_rag"]
    assert (r["estimate"], r["ci_low"], r["ci_high"]) == ("0.0", "0.0", "0.0")


def test_interval_is_the_2_5th_and_97_5th_percentile(tmp_path):
    # RAG+Laya - RAG+LLM resampled over 40 questions with 20 ones is 100 * Binomial(40, 0.5) / 40 pp.
    # Binomial(40, 0.5): P(X <= 13) = 0.0192 < 0.025 < P(X <= 14) = 0.0403 and
    # P(X <= 25) = 0.9597 < 0.975 < P(X <= 26) = 0.9808, so the 2.5th / 97.5th percentiles are
    # 14 and 26 questions = 35.0 and 65.0 pp (B = 10,000 keeps the empirical quantiles on these
    # values by more than four standard errors). A 5th / 95th or 15th / 85th interval would be narrower.
    assert run_script(write_run(tmp_path, CORRECT40), tmp_path / "out", b=10_000).returncode == 0
    r = rows(tmp_path / "out")["controller_rag"]
    assert (r["estimate"], r["ci_low"], r["ci_high"]) == ("50.0", "35.0", "65.0")


def test_seed_determines_the_output(tmp_path):
    # Small B makes the percentiles sensitive to the draws: the same seed reproduces the bytes,
    # a different seed changes them (so reproducibility comes from the seed, not from coarse data).
    runs = write_run(tmp_path, CORRECT40)
    out = {}
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        assert run_script(runs, tmp_path / name, b=20, seed=seed).returncode == 0
        out[name] = (tmp_path / name / "analysis" / "bootstrap_ci.csv").read_bytes()
    assert out["a"] == out["b"]
    assert out["a"] != out["c"]


def test_unequal_question_sets_exit_nonzero(tmp_path):
    # Pairing needs the same questions in all four conditions.
    runs = write_run(tmp_path, CORRECT40, qids={"rag_laya": N40[:39]})
    p = run_script(runs, tmp_path / "out")
    assert p.returncode != 0
    assert "same questions" in p.stdout + p.stderr
