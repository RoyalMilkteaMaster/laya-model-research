#!/usr/bin/env python3
"""Paired bootstrap 95% confidence intervals for the EM effects of run main-20260930.

Post-processing only (no model is run). Standard library only.

Input:  <runs>/<run_id>/records/*.jsonl and <runs>/<run_id>/summary.csv (read-only).
Rules: within one (dataset, model, condition) the last line of a question_id wins, as in
results_aggregator. A question whose final status is not "done" is also counted as EM 0; the
aggregator has no such rule (it sums the em field), but in run main-20260930 every non-done
record already carries em 0, so both give the same EM (checked against summary.csv below).

For every (dataset, model), the questions shared by the four conditions are resampled with
replacement (paired: the same draw is used for all four conditions), B times, with one
random.Random(SEED) consumed in the fixed order dataset (hotpotqa, 2wiki, musique) ->
model (27b ... 0.8b). Effects, in percentage points (pp):
  controller_rag    = EM(RAG+Laya)    - EM(RAG+LLM)
  controller_recall = EM(Recall+Laya) - EM(Recall+LLM)
  retrieval_llm     = EM(RAG+LLM)     - EM(Recall+LLM)
  interaction       = controller_rag  - controller_recall
The CI is the percentile interval: 2.5th / 97.5th percentile of the B replicates
(statistics.quantiles, method "inclusive", i.e. linear interpolation). All values one decimal.

Self-check: every point estimate must equal the same difference of the em column of
summary.csv, and n must equal summary.csv n; otherwise exit 1.

Outputs (byte-identical on every run):
  paper/analysis/bootstrap_ci.csv   dataset, model, effect, estimate, ci_low, ci_high, n, B, seed
  paper/tables/bootstrap_ci.tex     bare tabular (Ticket 11 contract); each data row ends with
                                    "% dataset=<d> model=<m> effects=..." naming its CSV rows;
                                    every number is copied verbatim from the CSV; missing "---".

Usage (from the Code Root):
  source multihop_benchmark/scripts/env.sh && uv run --project multihop_benchmark python paper/analysis/bootstrap_ci.py
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

SEED = 20261001
B = 10_000
DATASETS = ["hotpotqa", "2wiki", "musique"]
DATASET_LABELS = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}
MODELS = ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b"]
CONDITIONS = ["recall_llm", "rag_llm", "recall_laya", "rag_laya"]
# effect -> (minuend condition, subtrahend condition); interaction is derived.
PAIRS = {
    "controller_rag": ("rag_laya", "rag_llm"),
    "controller_recall": ("recall_laya", "recall_llm"),
    "retrieval_llm": ("rag_llm", "recall_llm"),
}
EFFECTS = ["controller_rag", "controller_recall", "retrieval_llm", "interaction"]
TABLE_EFFECTS = ["controller_rag", "controller_recall", "interaction"]
PAPER = Path(__file__).resolve().parent.parent


class CheckError(Exception):
    pass


def canonical_condition(value):
    return str(value or "").strip().lower().replace("+", "_").replace("-", "_")


def load_records(run_dir):
    """{(dataset, model, condition): {question_id: em (0/1)}}, last line per question wins."""
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
    return {k: {q: (int(float(it.get("em") or 0)) if it.get("status") == "done" else 0) for q, it in v.items()}
            for k, v in latest.items()}


def load_summary(run_dir):
    with open(run_dir / "summary.csv", newline="", encoding="utf-8") as f:
        return {(r["dataset"], r["model"], r["condition"]): r for r in csv.DictReader(f)}


def fmt(x):
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def effect_sums(vectors, idx):
    """Paired differences in correct-answer counts for the sampled question indices."""
    out = {e: sum(map(vectors[e].__getitem__, idx)) for e in PAIRS}
    out["interaction"] = out["controller_rag"] - out["controller_recall"]
    return out


def analyse(records, summary, b, seed):
    rng = random.Random(seed)
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            groups = [records.get((ds, model, c)) for c in CONDITIONS]
            if not all(groups):
                rows += [dict(dataset=ds, model=model, effect=e, estimate="", ci_low="", ci_high="", n="0", B=b, seed=seed)
                         for e in EFFECTS]
                continue
            qids = sorted(groups[0])
            if any(sorted(g) != qids for g in groups[1:]):
                raise CheckError(f"{ds}/{model}: the four conditions do not share the same questions (pairing impossible)")
            em = dict(zip(CONDITIONS, groups))
            n = len(qids)
            vectors = {e: [em[a][q] - em[s][q] for q in qids] for e, (a, s) in PAIRS.items()}
            point = effect_sums(vectors, range(n))
            reps = {e: [] for e in EFFECTS}
            for _ in range(b):
                sums = effect_sums(vectors, rng.choices(range(n), k=n))
                for e in EFFECTS:
                    reps[e].append(100 * sums[e] / n)
            for e in EFFECTS:
                cuts = statistics.quantiles(reps[e], n=40, method="inclusive")  # 2.5 % steps
                est = 100 * point[e] / n
                check_against_summary(summary, ds, model, e, est, n)
                rows.append(dict(dataset=ds, model=model, effect=e, estimate=fmt(est), ci_low=fmt(cuts[0]),
                                 ci_high=fmt(cuts[-1]), n=n, B=b, seed=seed))
    return rows


def check_against_summary(summary, ds, model, effect, est, n):
    def em(c):
        r = summary.get((ds, model, c))
        if r is None or int(r["n"]) != n:
            raise CheckError(f"summary.csv: {ds}/{model}/{c} missing or n != {n}")
        return float(r["em"])
    if effect == "interaction":
        expected = (em("rag_laya") - em("rag_llm")) - (em("recall_laya") - em("recall_llm"))
    else:
        a, s = PAIRS[effect]
        expected = em(a) - em(s)
    if fmt(expected) != fmt(est):
        raise CheckError(f"summary.csv: {ds}/{model} {effect} = {fmt(expected)} pp, records give {fmt(est)} pp")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "effect", "estimate", "ci_low", "ci_high", "n", "B", "seed"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def cell(r):
    if r["estimate"] == "":
        return "---"
    return f"${r['estimate']}$ $[{r['ci_low']}, {r['ci_high']}]$"


def write_table(rows, path):
    by = {(r["dataset"], r["model"], r["effect"]): r for r in rows}
    out = [
        "\\begin{tabular}{@{}l@{\\hspace{6pt}}c@{\\hspace{6pt}}c@{\\hspace{6pt}}c@{}}",
        "\\toprule",
        "Model & $\\Delta_{\\mathrm{ctrl}\\mid\\mathrm{RAG}}$ & $\\Delta_{\\mathrm{ctrl}\\mid\\mathrm{Recall}}$"
        " & $\\Delta_{\\mathrm{int}}$ \\\\",
    ]
    for ds in DATASETS:
        out += ["\\midrule", f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{DATASET_LABELS[ds]}}}}} \\\\"]
        for model in MODELS:
            cells = " & ".join(cell(by[(ds, model, e)]) for e in TABLE_EFFECTS)
            label = model.split(":")[1].upper()
            out.append(f"{label} & {cells} \\\\ % dataset={ds} model={model} effects={','.join(TABLE_EFFECTS)}")
    out += ["\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(PAPER.parent / "data" / "runs"))
    ap.add_argument("--run-id", default="main-20260930")
    ap.add_argument("--out-dir", default=str(PAPER / "analysis"))
    ap.add_argument("--table", default=str(PAPER / "tables" / "bootstrap_ci.tex"))
    ap.add_argument("-B", type=int, default=B)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    run_dir = Path(a.runs) / a.run_id
    try:
        rows = analyse(load_records(run_dir), load_summary(run_dir), a.B, a.seed)
    except CheckError as error:
        print(f"FAIL: {error}")
        return 1
    write_csv(rows, Path(a.out_dir) / "bootstrap_ci.csv")
    write_table(rows, Path(a.table))
    done = [r for r in rows if r["estimate"] != ""]
    excl = [r for r in done if float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0]
    print(f"OK: {len(done)} effects (B={a.B}, seed={a.seed}); point estimates match summary.csv; "
          f"{len(excl)} CIs exclude 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
