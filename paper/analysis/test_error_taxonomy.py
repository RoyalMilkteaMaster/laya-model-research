"""Tests for error_taxonomy.py on a small synthetic run (no model, no Data Root).

Run:  source multihop_benchmark/scripts/env.sh && uv run --project multihop_benchmark pytest paper/analysis
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("error_taxonomy.py")
GOLD = {"q0": ["g0a", "g0b"], "q1": ["g1a", "g1b"], "q2": ["g2a", "g2b"], "q3": ["g3a", "g3b"],
        "q4": ["g4a", "g4b"], "q5": ["g5a", "g5b"]}


def step(hop, sub, pids, p_suf=None):
    s = {"hop": hop, "sub_question": sub, "evidence_added": [{"pid": p} for p in pids],
         "llm": {"prompt_tokens": 1, "output_tokens": 1, "latency_ms": 1.0}}
    if p_suf is not None:
        s["laya"] = {"sufficient_p": p_suf, "next_action": "A" if p_suf >= 0.5 else "B", "remaining": 0}
    return s


DATASETS = ["hotpotqa", "2wiki", "musique"]


def rec(cond, q, em, status, steps):
    return {"dataset": "hotpotqa", "model": "qwen3.5:2b", "condition": cond, "question_id": q,
            "em": em, "status": status, "hops": len(steps), "steps": steps}


def make_run(tmp_path, summary_em=None):
    runs = tmp_path / "runs"
    run = runs / "main-20260930"
    (run / "records").mkdir(parents=True)
    for name in DATASETS:   # the same synthetic questions in every dataset, so pooled rows are 3x
        ds = tmp_path / "datasets" / name
        ds.mkdir(parents=True)
        ds.joinpath("sample_200.jsonl").write_text("".join(json.dumps(
            {"id": q, "paragraphs": [{"pid": p, "is_supporting": True} for p in g]
             + [{"pid": f"d{q}", "is_supporting": False}]}) + "\n" for q, g in GOLD.items()), encoding="utf-8")
    full = ["x", "y", "z"]
    laya = [
        rec("rag_laya", "q0", 1, "done", [step(1, "s", ["g0a", "g0b"], 0.1), step(2, "", [], 0.9)]),  # correct
        # all gold retrieved, still wrong
        rec("rag_laya", "q1", 0, "done", [step(1, "s", ["g1a"], 0.1), step(2, "s", ["g1b"], 0.2), step(3, "", [], 0.9)]),
        # Laya stopped with one gold paragraph missing
        rec("rag_laya", "q2", 0, "done", [step(1, "s", ["g2a", "dq2"], 0.1), step(2, "", [], 0.8)]),
        # four hops, gold never fully retrieved
        rec("rag_laya", "q3", 0, "done", [step(h, "s", full if h == 1 else [], 0.1) for h in (1, 2, 3, 4)]),
        # Laya stopped with nothing retrieved from gold, even at hop 4
        rec("rag_laya", "q4", 0, "done", [step(1, "s", ["x"], 0.1), step(2, "s", [], 0.1), step(3, "s", [], 0.1),
                                          step(4, "", [], 0.7)]),
        rec("rag_laya", "q5", 1, "done", [step(1, "s", ["g5a"], 0.1), step(2, "", [], 0.9)]),  # correct anyway
    ]
    llm = [
        rec("rag_llm", "q0", 0, "invalid_tool", [step(1, "", [])]),
        rec("rag_llm", "q1", 0, "timeout", [step(1, "s", ["g1a", "g1b"])]),   # timeout wins over coverage
        rec("rag_llm", "q2", 0, "error", [step(1, "s", ["g2a"]), step(2, "", [])]),
        # final_answer before H_max with gold missing: premature stop by the LLM controller
        rec("rag_llm", "q3", 0, "done", [step(1, "s", ["g3a"]), step(2, "", [])]),
        rec("rag_llm", "q4", 0, "done", []),  # no steps at all -> other
        # first written wrong, then redone correctly: the last line wins
        rec("rag_llm", "q5", 0, "done", [step(1, "s", ["g5a"]), step(2, "", [])]),
        rec("rag_llm", "q5", 1, "done", [step(1, "s", ["g5a", "g5b"]), step(2, "", [])]),
    ]
    for name in DATASETS:
        for cond, lines in (("rag_laya", laya), ("rag_llm", llm)):
            (run / "records" / f"{name}__qwen3.5-2b__{cond}.jsonl").write_text(
                "".join(json.dumps(dict(x, dataset=name)) + "\n" for x in lines), encoding="utf-8")
    em = {"rag_llm": "16.7", "rag_laya": "33.3"}
    em.update(summary_em or {})
    with open(run / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "condition", "n", "em"])
        for name in DATASETS:
            for c in ("recall_llm", "rag_llm", "recall_laya", "rag_laya"):
                w.writerow([name, "qwen3.5:2b", c, "6", em.get(c, "0.0")])
    return runs, tmp_path / "datasets"


def run_script(runs, datasets, out, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--runs", str(runs), "--datasets", str(datasets),
                           "--csv", str(out / "error_taxonomy.csv"), "--table", str(out / "error_taxonomy.tex"),
                           "--table-by-dataset", str(out / "error_taxonomy_by_dataset.tex"), *extra],
                          capture_output=True, text=True)


def counts(out, dataset="hotpotqa"):
    with open(out / "error_taxonomy.csv", encoding="utf-8") as f:
        return {(r["condition"], r["category"]): (int(r["count"]), r["share"]) for r in csv.DictReader(f)
                if (r["dataset"], r["model"]) == (dataset, "qwen3.5:2b")}


def test_each_wrong_answer_gets_exactly_one_category(tmp_path):
    runs, datasets = make_run(tmp_path)
    p = run_script(runs, datasets, tmp_path / "out")
    assert p.returncode == 0, p.stdout + p.stderr
    c = counts(tmp_path / "out")
    laya = {k[1]: v[0] for k, v in c.items() if k[0] == "rag_laya" and v[0]}
    llm = {k[1]: v[0] for k, v in c.items() if k[0] == "rag_llm" and v[0]}
    assert laya == {"evidence_covered": 1, "premature_stop": 2, "retrieval_miss": 1}
    assert llm == {"invalid_tool": 1, "timeout": 1, "error": 1, "premature_stop": 1, "other": 1}
    assert c[("rag_laya", "premature_stop")][1] == "0.500"   # share of the wrong answers in the cell


def test_pooled_rows_sum_over_datasets(tmp_path):
    runs, datasets = make_run(tmp_path)
    assert run_script(runs, datasets, tmp_path / "out").returncode == 0
    one, pooled = counts(tmp_path / "out"), counts(tmp_path / "out", "all")
    assert pooled == {k: (3 * n, share) for k, (n, share) in one.items()}


def assert_clean_failure(p, *needles):
    """Non-zero exit with a FAIL line naming the problem, never a traceback (finding R8 B-4)."""
    assert p.returncode == 1, p.stdout + p.stderr
    assert p.stdout.startswith("FAIL:") and "Traceback" not in p.stderr, p.stdout + p.stderr
    for n in needles:
        assert n in p.stdout, p.stdout


def test_missing_records_of_one_dataset_fail_instead_of_a_short_pooled_row(tmp_path):
    runs, datasets = make_run(tmp_path)
    (runs / "main-20260930" / "records" / "2wiki__qwen3.5-2b__rag_laya.jsonl").unlink()
    assert_clean_failure(run_script(runs, datasets, tmp_path / "out"), "2wiki/qwen3.5:2b/rag_laya")


def test_missing_inputs_fail_cleanly(tmp_path):
    runs, datasets = make_run(tmp_path)
    assert_clean_failure(run_script(tmp_path / "nowhere", datasets, tmp_path / "o1"), "summary.csv")
    (datasets / "musique" / "sample_200.jsonl").unlink()
    assert_clean_failure(run_script(runs, datasets, tmp_path / "o2"), "musique")


def test_missing_summary_column_fails_cleanly(tmp_path):
    runs, datasets = make_run(tmp_path)
    path = runs / "main-20260930" / "summary.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",em\n", ",exact\n", 1), encoding="utf-8")
    assert_clean_failure(run_script(runs, datasets, tmp_path / "out"), "em")


def test_wrong_count_must_match_summary(tmp_path):
    runs, datasets = make_run(tmp_path, summary_em={"rag_laya": "50.0"})
    p = run_script(runs, datasets, tmp_path / "out")
    assert p.returncode == 1
    assert "summary.csv" in p.stdout


def test_outputs_are_byte_identical_and_table_rows_name_their_csv_rows(tmp_path):
    runs, datasets = make_run(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_script(runs, datasets, a).returncode == 0
    assert run_script(runs, datasets, b).returncode == 0
    for name in ("error_taxonomy.csv", "error_taxonomy.tex"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    tex = (a / "error_taxonomy.tex").read_text(encoding="utf-8")
    assert tex.startswith("\\begin{tabular}") and tex.rstrip().endswith("\\end{tabular}")
    assert "\\caption" not in tex and "\\label" not in tex
    row = next(line for line in tex.splitlines() if "condition=rag_laya" in line and "qwen3.5:2b" in line)
    assert "dataset=all" in row
    # pooled counts of rag_laya over the three datasets: miss 3, covered 3, stop 6, in the header order
    assert row.split("%")[0].split("&")[2:] == [" 0 ", " 0 ", " 0 ", " 3 ", " 3 ", " 6 ", " 0 \\\\ "]


def test_default_table_style_is_full_grid_and_booktabs_is_kept_as_an_option(tmp_path):
    from test_merge_tables import assert_full_grid
    runs, datasets = make_run(tmp_path)
    assert run_script(runs, datasets, tmp_path / "g").returncode == 0
    out = tmp_path / "b"
    p = run_script(runs, datasets, out, "--style", "booktabs")
    assert p.returncode == 0, p.stdout + p.stderr
    grid = (tmp_path / "g" / "error_taxonomy.tex").read_text(encoding="utf-8")
    booktabs = (out / "error_taxonomy.tex").read_text(encoding="utf-8")
    assert_full_grid(grid)
    from test_merge_tables import assert_rows_match_columns
    assert_rows_match_columns(grid)
    assert_rows_match_columns((tmp_path / "g" / "error_taxonomy_by_dataset.tex").read_text(encoding="utf-8"))
    assert "\\toprule" in booktabs and "|" not in booktabs.replace("\\\\", "")
    data = lambda t: [x for x in t.splitlines() if "% dataset=" in x]
    assert data(grid) == data(booktabs) and data(grid)
    assert (tmp_path / "g" / "error_taxonomy.csv").read_bytes() == (out / "error_taxonomy.csv").read_bytes()


def test_by_dataset_table_lists_every_dataset_model_and_condition_verbatim(tmp_path):
    """R9 round 2: appendix table of the per-dataset counts (the main-text table pools the datasets)."""
    from test_merge_tables import assert_full_grid
    runs, datasets = make_run(tmp_path)
    out = tmp_path / "o"
    assert run_script(runs, datasets, out).returncode == 0
    tex = (out / "error_taxonomy_by_dataset.tex").read_text(encoding="utf-8")
    assert_full_grid(tex)
    rows = [x for x in tex.splitlines() if "% dataset=" in x]
    assert [x.split("% ")[1] for x in rows] == [f"dataset={d} model=qwen3.5:2b condition={c}"
                                                 for d in DATASETS for c in ("rag_llm", "rag_laya")]
    with open(out / "error_taxonomy.csv", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    for x in rows:
        d, c = x.split("dataset=")[1].split()[0], x.split("condition=")[1].split()[0]
        want = [r["count"] for k in ("invalid_tool", "timeout", "error", "retrieval_miss", "evidence_covered",
                                      "premature_stop", "other")
                for r in csv_rows if (r["dataset"], r["condition"], r["category"]) == (d, c, k)]
        assert [v.strip() for v in x.split("%")[0].rstrip().rstrip("\\").split("&")[3:]] == want
    a = run_script(runs, datasets, tmp_path / "b")
    assert a.returncode == 0 and (tmp_path / "b" / "error_taxonomy_by_dataset.tex").read_bytes() == tex.encode("utf-8")


def test_by_dataset_table_defaults_next_to_the_pooled_table(tmp_path):
    """R9 round 3 (Reviewer B, B-5): without --table-by-dataset the per-dataset table is written next to --table."""
    runs, datasets = make_run(tmp_path)
    out = tmp_path / "o"
    p = subprocess.run([sys.executable, str(SCRIPT), "--runs", str(runs), "--datasets", str(datasets),
                        "--csv", str(out / "error_taxonomy.csv"), "--table", str(out / "error_taxonomy.tex")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "error_taxonomy_by_dataset.tex").is_file()
