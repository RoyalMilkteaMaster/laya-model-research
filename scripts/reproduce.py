#!/usr/bin/env python3
"""Verify the frozen run and regenerate its statistics and figures without a GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "multihop_benchmark"))
sys.path.insert(0, str(ROOT / "paper" / "analysis"))

from multihop_benchmark.evaluation.answer_scorer import exact_match, f1
from multihop_benchmark.reporting import results_aggregator as agg
import bootstrap_ci as ci
import error_taxonomy as taxonomy
import merge_tables as panels

DATA = ROOT / "data"
RUN = DATA / "runs" / "main-20260930"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_checksums():
    checked = 0
    for line in (ROOT / "provenance" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), f"Unsafe checksum path: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == digest, f"Checksum mismatch: {name}")
        checked += 1
    require(checked > 0, "Empty checksum manifest")
    return checked


def verify_records(records, rows):
    require(len(records) == 12000, f"Expected 12,000 final records, got {len(records)}")
    require(len(rows) == 60 and all(r["n"] == "200" for r in rows), "Expected 60 cells of 200 questions")
    samples = {}
    for ds in agg.DATASETS:
        questions = [json.loads(line) for line in (DATA / "datasets" / ds / "sample_200.jsonl").read_text(encoding="utf-8").splitlines()]
        samples[ds] = {str(q["id"]): q for q in questions}
        require(len(samples[ds]) == 200, f"{ds}: duplicate or missing question IDs")
    ids = defaultdict(set)
    for record in records:
        ds, model, condition = (record[k] for k in ("dataset", "model", "condition"))
        qid = str(record["question_id"])
        require(qid in samples[ds], f"Unknown question: {ds}/{qid}")
        question = samples[ds][qid]
        require(record["gold"] == question["answer"] and record["gold_aliases"] == question["answer_aliases"],
                f"Gold answer differs from the frozen sample: {ds}/{qid}")
        ids[(ds, model, condition)].add(qid)
        expected_em, expected_f1 = 0, 0.0
        if record["status"] == "done":
            expected_em = exact_match(record["prediction"], record["gold"], record["gold_aliases"])
            expected_f1 = f1(record["prediction"], record["gold"], record["gold_aliases"])
        require(record["em"] == expected_em and math.isclose(record["f1"], expected_f1, abs_tol=1e-12),
                f"Answer score mismatch: {ds}/{model}/{condition}/{qid}")
    for (ds, model, condition), found in ids.items():
        require(found == set(samples[ds]), f"Question pairing mismatch: {ds}/{model}/{condition}")


def verify_generated(output):
    comparisons = [(output / "summary.csv", RUN / "summary.csv"),
                   (output / "analysis/bootstrap_ci.csv", ROOT / "paper/analysis/bootstrap_ci.csv"),
                   (output / "analysis/error_taxonomy.csv", ROOT / "paper/analysis/error_taxonomy.csv")]
    for generated, reference in comparisons:
        require(read_csv(generated) == read_csv(reference), f"Recomputed values differ: {reference.name}")
    tables = sorted((output / "tables").glob("*.tex"))
    for generated in tables:
        reference = ROOT / "paper/tables" / generated.name
        require(generated.read_text(encoding="utf-8") == reference.read_text(encoding="utf-8"),
                f"Recomputed table differs: {reference.name}")
    return len(tables)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/reproduced")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output != ROOT and not ROOT.is_relative_to(output), "Output must not overwrite the repository root")
    for protected in (DATA, ROOT / "paper", ROOT / "provenance", ROOT / "multihop_benchmark", ROOT / "results"):
        require(not output.is_relative_to(protected), f"Output must not overwrite frozen inputs: {protected.name}")
    verified_files = verify_checksums()
    records, skipped = agg.load_records(RUN / "records")
    require(skipped == 0, "Frozen records contain an incomplete line")
    rows = agg.summarize(records)
    verify_records(records, rows)
    require(rows == read_csv(RUN / "summary.csv"), "Recomputed summary differs from the paper run")
    output.mkdir(parents=True, exist_ok=True)
    agg.write_summary_csv(rows, output / "summary.csv")
    agg.write_tables(rows, output / "tables")
    agg.write_probe_table(RUN / "probe_results.json", output / "tables")
    print("PASS: checksums, 12,000 answer scores, matched question IDs, and all 60 summary rows", flush=True)
    effects = ci.analyse(ci.load_records(RUN), ci.load_summary(RUN), ci.B, ci.SEED)
    ci.write_csv(effects, output / "analysis/bootstrap_ci.csv")
    ci.write_table(effects, output / "tables/bootstrap_ci.tex")
    errors = taxonomy.analyse(taxonomy.load_records(RUN), taxonomy.load_gold(DATA / "datasets"), taxonomy.load_summary(RUN))
    taxonomy.write_csv(errors, output / "analysis/error_taxonomy.csv")
    taxonomy.write_table(errors, output / "tables/error_taxonomy.tex")
    taxonomy.write_table_by_dataset(errors, output / "tables/error_taxonomy_by_dataset.tex")
    require(panels.main(["--runs", str(DATA / "runs"), "--ci", str(output / "analysis/bootstrap_ci.csv"),
                         "--tables", str(output / "tables")]) == 0, "Panel generation failed")
    table_count = verify_generated(output)
    figures = agg.build_figures(rows, records)
    (output / "figures").mkdir(exist_ok=True)
    for stem, figure in figures.items():
        figure.savefig(output / "figures" / f"{stem}.pdf", metadata={"CreationDate": None})
        figure.savefig(output / "figures" / f"{stem}.png", dpi=180)
        figure.clear()
    import matplotlib
    import pandas
    report = {
        "status": "PASS", "run_id": RUN.name, "verified_checksum_files": verified_files,
        "final_records": len(records), "configurations": len(rows), "questions_per_configuration": 200,
        "answer_scores_recomputed": len(records), "summary_matches": True,
        "bootstrap_effects": len(effects), "bootstrap_matches": True,
        "error_taxonomy_rows": len(errors), "error_taxonomy_matches": True,
        "matching_latex_tables": table_count, "figures_regenerated": len(figures),
        "final_status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__, "pandas": pandas.__version__,
        "scope": "Re-analysis of saved answers and traces; no model inference, retraining, or new latency measurements.",
        "table_comparison": "Exact text after normalizing platform line endings; CSV cells compared exactly.",
        "controller_evaluation": "The historical aggregate evaluation report is an input; individual controller predictions were not archived.",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(effects)} bootstrap effects, {len(errors)} error rows, {table_count} tables, {len(figures)} figures")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
