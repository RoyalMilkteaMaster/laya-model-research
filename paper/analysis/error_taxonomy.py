#!/usr/bin/env python3
"""Error taxonomy of the wrong answers under retrieval (RAG+LLM, RAG+Laya) of run main-20260930.

Post-processing only (no model is run). Standard library only.

Inputs (read-only): <runs>/<run_id>/records/*.jsonl, <runs>/<run_id>/summary.csv, and
<datasets>/<dataset>/sample_200.jsonl (gold supporting paragraphs: paragraphs[].is_supporting).
Within one (dataset, model, condition) the last line of a question_id wins, as in
results_aggregator. A question is wrong if its final status is not "done" or its em is 0.

Every wrong answer gets exactly one category, checked in this order:
  invalid_tool / timeout / error   final status (any other non-done status, e.g. oom -> error)
  evidence_covered   every gold supporting paragraph was retrieved in some hop (union of the
                     passage ids in steps[].evidence_added), yet the answer is wrong
  premature_stop     a gold paragraph is missing and the controller ended the loop: the last
                     step is a stop (empty sub_question), i.e. Laya's p_suf >= 0.5 or the LLM's
                     final_answer, at any hop up to H_max
  retrieval_miss     a gold paragraph is missing although the loop used every hop (the last step
                     still asked a sub-question and retrieved)
  other              anything else (e.g. a done record without steps)

share = count / wrong answers of the (dataset, model, condition), three decimals. Rows with
dataset "all" pool the three datasets (counts summed, share recomputed).
Self-check: the number of wrong answers must equal n * (1 - em / 100) of summary.csv; else exit 1.
Missing data never shrinks a row silently (R8 finding B-4): a (dataset, model, condition) that
summary.csv lists without records, a model and condition whose records do not cover all three
datasets (the pooled row would hold fewer questions), a missing summary.csv or sample_200.jsonl,
and a missing field all exit 1 with a "FAIL:" line and no traceback.

Outputs (byte-identical on every run):
  paper/analysis/error_taxonomy.csv   dataset, model, condition, category, count, share
  paper/tables/error_taxonomy.tex     bare tabular (Ticket 11 contract) of the pooled counts; each
                                      row ends with "% dataset=all model=<m> condition=<c>"; every
                                      number is copied verbatim from the CSV. --style grid (default,
                                      revision R9) rules every cell as in the reference paper;
                                      --style booktabs reproduces the R8 booktabs tabular.
  paper/tables/error_taxonomy_by_dataset.tex  appendix B (R9 round 2): the same counts per dataset
                                      (rows "% dataset=<d> model=<m> condition=<c>"), same style;
                                      written next to --table unless --table-by-dataset is given.

Usage (from the Code Root):
  source multihop_benchmark/scripts/env.sh && uv run --project multihop_benchmark python paper/analysis/error_taxonomy.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DATASETS = ["hotpotqa", "2wiki", "musique"]
MODELS = ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b"]
CONDITIONS = ["rag_llm", "rag_laya"]
CONDITION_LABELS = {"rag_llm": "RAG+LLM", "rag_laya": "RAG+Laya"}
CATEGORIES = ["invalid_tool", "timeout", "error", "retrieval_miss", "evidence_covered", "premature_stop", "other"]
HEADERS = ["Inv.", "T/O", "Err.", "Miss", "Cov.", "Stop", "Oth."]
PAPER = Path(__file__).resolve().parent.parent


class CheckError(Exception):
    pass


def canonical_condition(value):
    return str(value or "").strip().lower().replace("+", "_").replace("-", "_")


def load_records(run_dir):
    """{(dataset, model, condition): {question_id: record}}, last line per question wins."""
    latest = {}
    for path in sorted((run_dir / "records").glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").split("\n")
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines):  # last line without newline: the run was cut mid-write
                    continue
                raise CheckError(f"corrupt record {path.name}:{number}")
            key = (str(item.get("dataset", "")).strip().lower(), str(item.get("model", "")).strip().lower(),
                   canonical_condition(item.get("condition")))
            latest.setdefault(key, {})[str(item.get("question_id"))] = item
    return latest


def load_gold(datasets_dir):
    gold = {}
    for ds in DATASETS:
        path = datasets_dir / ds / "sample_200.jsonl"
        if not path.exists():
            raise CheckError(f"gold paragraphs missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                gold[(ds, str(item["id"]))] = {p["pid"] for p in item["paragraphs"] if p.get("is_supporting")}
    return gold


def load_summary(run_dir):
    path = run_dir / "summary.csv"
    if not path.exists():
        raise CheckError(f"summary.csv missing: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("dataset", "model", "condition", "n", "em") if rows and c not in rows[0]]
    if missing:
        raise CheckError(f"summary.csv: missing column(s) {', '.join(missing)}")
    return {(r["dataset"], r["model"], r["condition"]): r for r in rows}


def is_wrong(item):
    return item.get("status") != "done" or not int(float(item.get("em") or 0))


def category(item, gold):
    status = item.get("status")
    if status in ("invalid_tool", "timeout"):
        return status
    if status != "done":
        return "error"
    steps = item.get("steps") or []
    if not steps:
        return "other"
    retrieved = {e.get("pid") for s in steps for e in (s.get("evidence_added") or [])}
    if gold <= retrieved:
        return "evidence_covered"
    if not steps[-1].get("sub_question"):
        return "premature_stop"
    return "retrieval_miss"


def share(count, wrong):
    return f"{count / wrong:.3f}" if wrong else "0.000"


def analyse(records, gold, summary):
    rows, pooled, covered = [], {}, {}
    for ds in DATASETS:
        for model in MODELS:
            for cond in CONDITIONS:
                items = records.get((ds, model, cond))
                if not items:
                    if (ds, model, cond) in summary:
                        raise CheckError(f"records missing for {ds}/{model}/{cond} (listed in summary.csv)")
                    continue
                covered.setdefault((model, cond), []).append(ds)
                counts = dict.fromkeys(CATEGORIES, 0)
                for qid, item in sorted(items.items()):
                    if not is_wrong(item):
                        continue
                    if (ds, qid) not in gold:
                        raise CheckError(f"{ds}/{qid}: no gold paragraphs in sample_200.jsonl")
                    counts[category(item, gold[(ds, qid)])] += 1
                wrong = sum(counts.values())
                check_against_summary(summary, ds, model, cond, len(items), wrong)
                for c in CATEGORIES:
                    rows.append(dict(dataset=ds, model=model, condition=cond, category=c, count=counts[c],
                                     share=share(counts[c], wrong)))
                    pooled.setdefault((model, cond), dict.fromkeys(CATEGORIES, 0))[c] += counts[c]
    for (model, cond), present in covered.items():
        absent = [ds for ds in DATASETS if ds not in present]
        if absent:
            raise CheckError(f"records missing for {', '.join(f'{ds}/{model}/{cond}' for ds in absent)}: "
                             f"the pooled dataset=all row needs all of {', '.join(DATASETS)}")
    for model in MODELS:
        for cond in CONDITIONS:
            counts = pooled.get((model, cond))
            if counts:
                wrong = sum(counts.values())
                rows += [dict(dataset="all", model=model, condition=cond, category=c, count=counts[c],
                              share=share(counts[c], wrong)) for c in CATEGORIES]
    return rows


def check_against_summary(summary, ds, model, cond, n, wrong):
    r = summary.get((ds, model, cond))
    if r is None or int(r["n"]) != n:
        raise CheckError(f"summary.csv: {ds}/{model}/{cond} missing or n != {n}")
    expected = round(n * (1 - float(r["em"]) / 100))
    if expected != wrong:
        raise CheckError(f"summary.csv: {ds}/{model}/{cond} implies {expected} wrong answers, records give {wrong}")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "condition", "category", "count", "share"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_table(rows, path, style="grid"):
    by = {(r["dataset"], r["model"], r["condition"], r["category"]): r for r in rows}
    ncols = 2 + len(CATEGORIES)
    if style == "grid":   # every column ruled; \\cline inside a model group, \\hline after it
        out = ["\\begin{tabular}{|" + "|".join(a for a in "ll" + "r" * len(CATEGORIES)) + "|}",
               "\\hline", "Model & Condition & " + " & ".join(HEADERS) + " \\\\"]
    else:
        sep = "@{\\hspace{4pt}}"
        out = [f"\\begin{{tabular}}{{@{{}}l{sep}l{sep}" + sep.join("r" * len(CATEGORIES)) + "@{}}",
               "\\toprule",
               "Model & Condition & " + " & ".join(HEADERS) + " \\\\"]
    for model in MODELS:
        present = [c for c in CONDITIONS if ("all", model, c, CATEGORIES[0]) in by]
        if not present:
            continue
        out.append("\\hline" if style == "grid" else "\\midrule")
        for i, cond in enumerate(present):
            label = model.split(":")[1].upper() if i == 0 else ""
            cells = " & ".join(str(by[("all", model, cond, c)]["count"]) for c in CATEGORIES)
            if style == "grid" and i > 0:
                out.append(f"\\cline{{2-{ncols}}}")
            out.append(f"{label} & {CONDITION_LABELS[cond]} & {cells} \\\\ % dataset=all model={model} condition={cond}")
    out += ["\\hline" if style == "grid" else "\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


DATASET_LABELS = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}


def write_table_by_dataset(rows, path, style="grid"):
    """Appendix table (R9 round 2): counts per dataset, model, and condition, copied from the CSV rows."""
    by = {(r["dataset"], r["model"], r["condition"], r["category"]): r for r in rows}
    ncols = 3 + len(CATEGORIES)
    head = "Dataset & Model & Condition & " + " & ".join(HEADERS) + " \\\\"
    if style == "grid":
        out = ["\\begin{tabular}{|" + "|".join(a for a in "lll" + "r" * len(CATEGORIES)) + "|}",
               "\\hline", head]
    else:
        out = ["\\begin{tabular}{@{}lll" + "r" * len(CATEGORIES) + "@{}}", "\\toprule", head]
    for ds in DATASETS:
        out.append("\\hline" if style == "grid" else "\\midrule")
        models = [m for m in MODELS if (ds, m, CONDITIONS[0], CATEGORIES[0]) in by]
        for j, model in enumerate(models):
            if j > 0:
                out.append("\\cline{2-%d}" % ncols if style == "grid" else "\\cmidrule(l){2-%d}" % ncols)
            for i, cond in enumerate(c for c in CONDITIONS if (ds, model, c, CATEGORIES[0]) in by):
                if style == "grid" and i > 0:
                    out.append(f"\\cline{{3-{ncols}}}")
                labels = [DATASET_LABELS[ds] if j == 0 and i == 0 else "", model.split(":")[1].upper() if i == 0 else ""]
                cells = " & ".join(str(by[(ds, model, cond, c)]["count"]) for c in CATEGORIES)
                out.append(" & ".join(labels) + f" & {CONDITION_LABELS[cond]} & {cells} \\\\ % dataset={ds} model={model} condition={cond}")
    out += ["\\hline" if style == "grid" else "\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(PAPER.parent / "data" / "runs"))
    ap.add_argument("--run-id", default="main-20260930")
    ap.add_argument("--datasets", default=str(PAPER.parent / "data" / "datasets"))
    ap.add_argument("--csv", default=str(PAPER / "analysis" / "error_taxonomy.csv"))
    ap.add_argument("--table", default=str(PAPER / "tables" / "error_taxonomy.tex"))
    ap.add_argument("--table-by-dataset", default=None,
                    help="default: error_taxonomy_by_dataset.tex next to --table (R9 round 3)")
    ap.add_argument("--style", choices=["grid", "booktabs"], default="grid",
                    help="grid: ruled cells as in the reference paper (R9, default); booktabs: the R8 table")
    a = ap.parse_args(argv)
    if a.table_by_dataset is None:
        a.table_by_dataset = str(Path(a.table).with_name("error_taxonomy_by_dataset.tex"))
    run_dir = Path(a.runs) / a.run_id
    try:
        summary = load_summary(run_dir)
        rows = analyse(load_records(run_dir), load_gold(Path(a.datasets)), summary)
    except CheckError as error:
        print(f"FAIL: {error}")
        return 1
    except KeyError as error:
        print(f"FAIL: missing field {error} in the records, summary.csv, or sample_200.jsonl")
        return 1
    except (OSError, ValueError) as error:   # unreadable file, malformed JSON / number
        print(f"FAIL: {error}")
        return 1
    write_csv(rows, Path(a.csv))
    write_table(rows, Path(a.table), a.style)
    write_table_by_dataset(rows, Path(a.table_by_dataset), a.style)
    wrong = sum(int(r["count"]) for r in rows if r["dataset"] != "all")
    print(f"OK: {wrong} wrong answers in {len(rows) // len(CATEGORIES)} cells classified; "
          f"wrong counts match summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
