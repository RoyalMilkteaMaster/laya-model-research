#!/usr/bin/env python3
"""Panel tables of revision R8: tables merged so that every table cited in the main text fits there.

Post-processing only (no model is run). Standard library only. The older tables in paper/tables/
(main_results, efficiency_*, bootstrap_ci, probe) are left untouched.

Inputs (read-only):
  <runs>/<run_id>/summary.csv                  EM, F1, efficiency columns
  paper/analysis/bootstrap_ci.csv              paired bootstrap effects (analysis/bootstrap_ci.py)
  <runs>/<run_id>/probe_results.json           function-calling probe
  <runs>/laya_eval/laya_eval_report.md         decision-layer quality (table "指標" and the 整體 line)

Outputs (bare tabulars, Ticket 11 contract; byte-identical on every run):
  paper/tables/main_results_panels.tex  Panel A: EM / F1 per model x condition x dataset (summary.csv);
                                        Panel B: estimate [CI] (bootstrap_ci.csv); grid style (R9 round 2):
                                        controller_rag and interaction, one row per model and dataset;
                                        booktabs style (R8): the three effects, one column per dataset
  paper/tables/efficiency_panels.tex    Panels A/B/C (HotpotQA, 2Wiki, MuSiQue) side by side: p50
                                        latency, VRAM, In / Out tokens, hops, Laya ms (summary.csv)
  paper/tables/controller_probe.tex     Panel A: probe items, valid rate, mean latency
                                        (probe_results.json, formatted as results_aggregator does:
                                        int, two decimals, no decimals);
                                        Panel B: decision-layer quality (laya_eval_report.md)
  paper/tables/effects_recall_retrieval.tex  appendix B (R9 round 2): controller_recall and retrieval_llm,
                                        estimate [CI] per model and dataset (bootstrap_ci.csv)
  paper/tables/efficiency_p95.tex       appendix B (R9 round 2): latency_p95_ms per model, condition, and
                                        dataset (summary.csv)
Table style (revision R9): --style grid (default) rules every column and follows every row with \\hline
(\\cline within a model group), as in the tables of the reference paper (IEEE AVSS 2025, CVPR kit);
--style booktabs reproduces the R8 booktabs tabulars byte for byte. Both styles hold the same rows,
except Panel B of main_results_panels (see above).
Every summary.csv / bootstrap_ci.csv / laya_eval_report.md cell is copied verbatim (empty or
"—" -> "---"); each data row ends with "% key=value" naming its source row. A missing input
row is an error (exit 1), never a silent gap.

Usage (from the Code Root):
  source multihop_benchmark/scripts/env.sh && uv run --project multihop_benchmark python paper/analysis/merge_tables.py
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

DATASETS = ["hotpotqa", "2wiki", "musique"]
DATASET_LABELS = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}
MODELS = ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b"]
CONDITIONS = ["recall_llm", "rag_llm", "recall_laya", "rag_laya"]
CONDITION_LABELS = {"recall_llm": "Recall+LLM", "rag_llm": "RAG+LLM", "recall_laya": "Recall+Laya",
                    "rag_laya": "RAG+Laya"}
EFFECTS = {"controller_rag": "$\\Delta_{\\mathrm{RAG}}$", "controller_recall": "$\\Delta_{\\mathrm{Rec}}$",
           "interaction": "$\\Delta_{\\mathrm{int}}$"}
# R9 round 2: the grid Panel B keeps these two effects (one row per model and dataset); the recall
# effect and the retrieval effect (RAG+LLM minus Recall+LLM) go to the appendix table.
EFFECTS_MAIN = ["controller_rag", "interaction"]
EFFECTS_APPENDIX = {"controller_recall": "$\\Delta_{\\mathrm{Rec}}$", "retrieval_llm": "$\\Delta_{\\mathrm{Ret}}$"}
EFFICIENCY = ["latency_p50_ms", "vram_peak_mb", "prompt_tokens_mean", "output_tokens_mean", "hops_mean",
              "laya_latency_ms_mean"]
EFFICIENCY_HEADERS = ["p50", "VRAM", "In", "Out", "Hops", "Laya"]
LAYA_QUESTIONS = ["sufficient", "next_action", "remaining_hops"]
LAYA_MODELS = {"微調": "FT", "zero-shot": "ZS"}
MISSING = "---"
PAPER = Path(__file__).resolve().parent.parent


class CheckError(Exception):
    pass


def model_label(model):
    return model.split(":")[1].upper()


def verbatim(value):
    value = (value or "").strip()
    return MISSING if value in ("", "—") else value


def load_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = {(r["dataset"], r["model"], r["condition"]): r for r in csv.DictReader(f)}
    for ds in DATASETS:
        for model in MODELS:
            for cond in CONDITIONS:
                if (ds, model, cond) not in rows:
                    raise CheckError(f"summary.csv: no row {ds}/{model}/{cond}")
    return rows


def load_ci(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = {(r["dataset"], r["model"], r["effect"]): r for r in csv.DictReader(f)}
    for ds in DATASETS:
        for model in MODELS:
            for effect in list(EFFECTS) + list(EFFECTS_APPENDIX):
                if (ds, model, effect) not in rows:
                    raise CheckError(f"bootstrap_ci.csv: no row {ds}/{model}/{effect}")
    return rows


def load_probe(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", data)
    for model in MODELS:
        if model not in models:
            raise CheckError(f"probe_results.json: no model {model}")
    return models


def load_laya(path):
    """{(question, model label): {accuracy, Brier, ECE, MAE}} and {model label: (accuracy, ECE)}."""
    rows, header = {}, None
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[:2] == ["問題", "模型"]:
            header = cells
        elif header and len(cells) == len(header) and cells[0] in LAYA_QUESTIONS and cells[1] in LAYA_MODELS:
            rows[(cells[0], cells[1])] = dict(zip(header, cells))
    m = re.search(r"整體 accuracy：微調 ([\d.]+)、zero-shot ([\d.]+)；整體 ECE：微調 ([\d.]+)、zero-shot ([\d.]+)", text)
    if not m:
        raise CheckError("laya_eval_report.md: no 整體 accuracy / ECE line")
    for q in LAYA_QUESTIONS:
        for model in LAYA_MODELS:
            if (q, model) not in rows:
                raise CheckError(f"laya_eval_report.md: no row {q}/{model}")
    return rows, {"微調": (m.group(1), m.group(3)), "zero-shot": (m.group(2), m.group(4))}


def row(cells, comment):
    return " & ".join(cells) + f" \\\\ % {comment}"


# Grid style: no @{} padding next to the rules (R9 round 3): the cell padding is \tabcolsep, which the hook
# macros of main.tex set for the generated tables, so that every \cline and \multicolumn meets the rules.


def grid_spec(aligns):
    """Column spec with a vertical rule at both edges and between all columns: |l|r|..."""
    return "|" + "|".join(aligns) + "|"


def grid_group(rows_, ncols):
    """Rows of one model group in grid style: \\cline under all but the label column inside the
    group, \\hline after its last row."""
    out = []
    for i, line in enumerate(rows_):
        out += [line, "\\hline" if i == len(rows_) - 1 else f"\\cline{{2-{ncols}}}"]
    return out


def panel_a_rows(summary, model):
    out = []
    for i, cond in enumerate(CONDITIONS):
        cells = [model_label(model) if i == 0 else "", CONDITION_LABELS[cond]]
        for ds in DATASETS:
            r = summary[(ds, model, cond)]
            cells += [verbatim(r["em"]), verbatim(r["f1"])]
        out.append(row(cells, f"model={model} condition={cond}"))
    return out


def ci_cell(r):
    """estimate [ci_low, ci_high], every number verbatim from bootstrap_ci.csv."""
    if verbatim(r["estimate"]) == MISSING:
        return MISSING
    return f"${r['estimate']}$\\,$[{r['ci_low']}, {r['ci_high']}]$"


def panel_b_rows(ci, model):
    out = []
    for i, (effect, label) in enumerate(EFFECTS.items()):
        cells = [model_label(model) if i == 0 else "", label]
        cells += [ci_cell(ci[(ds, model, effect)]) for ds in DATASETS]
        out.append(row(cells, f"model={model} effect={effect} datasets={','.join(DATASETS)}"))
    return out


def effect_rows_by_dataset(ci, model, effects):
    """One row per dataset: model label (first row only), dataset, one estimate [CI] cell per effect."""
    return [row([model_label(model) if i == 0 else "", DATASET_LABELS[ds]] + [ci_cell(ci[(ds, model, e)]) for e in effects],
                f"model={model} dataset={ds} effects={','.join(effects)}")
            for i, ds in enumerate(DATASETS)]


def simple_table(aligns, head, groups, style):
    """A grid (R9) or booktabs table: header row(s), then groups of data rows."""
    ncols = len(aligns)
    if style == "grid":
        out = [f"\\begin{{tabular}}{{{grid_spec(aligns)}}}", "\\hline"]
        for line in head:
            out += [line, "\\hline"]
        for group in groups:
            out += grid_group(group, ncols)
        return out + ["\\end{tabular}"]
    out = [f"\\begin{{tabular}}{{@{{}}{aligns}@{{}}}}", "\\toprule"] + head
    for group in groups:
        out += ["\\midrule"] + group
    return out + ["\\bottomrule", "\\end{tabular}"]


def effects_appendix(ci, style="grid"):
    """Appendix table (R9 round 2): recall effect and retrieval effect with paired bootstrap CIs."""
    head = ["Model & Dataset & " + " & ".join(f"{label} [95\\% CI]" for label in EFFECTS_APPENDIX.values()) + " \\\\"]
    return simple_table("ll" + "c" * len(EFFECTS_APPENDIX), head,
                        [effect_rows_by_dataset(ci, m, list(EFFECTS_APPENDIX)) for m in MODELS], style)


def efficiency_p95(summary, style="grid"):
    """Appendix table (R9 round 2): end-to-end p95 latency (ms) per model, condition, and dataset."""
    head = ["Model & Condition & " + " & ".join(DATASET_LABELS[d] for d in DATASETS) + " \\\\"]
    groups = [[row([model_label(m) if i == 0 else "", CONDITION_LABELS[c]]
                   + [verbatim(summary[(ds, m, c)]["latency_p95_ms"]) for ds in DATASETS],
                   f"model={m} condition={c} datasets={','.join(DATASETS)}")
               for i, c in enumerate(CONDITIONS)] for m in MODELS]
    return simple_table("ll" + "r" * len(DATASETS), head, groups, style)


# R9 round 3 (Reviewer B, B-1): the grid panels of main_results_panels are plain tabulars with fixed
# column widths (array package, loaded by main.tex): the Model and label columns have the same width in
# both panels and the data columns share the rest of \columnwidth, so both panels are column-wide and
# every rule meets its \cline. (tabular* with \extracolsep put the stretch glue next to the rules.)
PANEL_FIXED = {"model": 19, "label": 37}   # pt; widest entries at 7 pt: "Model", "Recall+Laya"


def panel_spec(ndata, align):
    """|Model|label|data...| with fixed widths; the ndata data columns fill the rest of \\columnwidth."""
    ncols = 2 + ndata
    fixed = PANEL_FIXED["model"] + PANEL_FIXED["label"]
    data = (f"p{{\\dimexpr(\\columnwidth-{fixed}pt-{2 * ncols}\\tabcolsep-{ncols + 1}\\arrayrulewidth)"
            f"/{ndata}\\relax}}")
    cols = [f">{{\\raggedright\\arraybackslash}}p{{{PANEL_FIXED['model']}pt}}",
            f">{{\\raggedright\\arraybackslash}}p{{{PANEL_FIXED['label']}pt}}"] + [f">{{{align}\\arraybackslash}}{data}"] * ndata
    return grid_spec(cols)


def main_results_panels(summary, ci, style="grid"):
    """Panel A and Panel B are separate column-wide tables stacked in one outer tabular, so that
    the wide estimate [CI] cells of Panel B do not widen the EM / F1 columns of Panel A."""
    if style == "grid":
        out = ["\\begin{tabular}{@{}c@{}}",
               "\\begin{tabular}{" + panel_spec(6, "\\raggedleft") + "}",
               "\\hline",
               "\\multicolumn{8}{|l|}{\\emph{Panel A: EM / F1 (\\%)}} \\\\",
               "\\hline",
               "\\multicolumn{2}{|c|}{} & "
               + " & ".join(f"\\multicolumn{{2}}{{c|}}{{{DATASET_LABELS[d]}}}" for d in DATASETS) + " \\\\",
               "\\cline{3-8}",
               "Model & Condition & EM & F1 & EM & F1 & EM & F1 \\\\",
               "\\hline"]
        for model in MODELS:
            out += grid_group(panel_a_rows(summary, model), 8)
        out += ["\\end{tabular} \\\\",
                "\\begin{tabular}{" + panel_spec(len(EFFECTS_MAIN), "\\centering") + "}",
                f"\\multicolumn{{{2 + len(EFFECTS_MAIN)}}}{{|l|}}{{\\emph{{Panel B: EM effect (pp), estimate [paired bootstrap 95\\% CI]}}}} \\\\",
                "\\hline",
                "Model & Dataset & " + " & ".join(EFFECTS[e] for e in EFFECTS_MAIN) + " \\\\",
                "\\hline"]
        for model in MODELS:
            out += grid_group(effect_rows_by_dataset(ci, model, EFFECTS_MAIN), 2 + len(EFFECTS_MAIN))
        return out + ["\\end{tabular}", "\\end{tabular}"]
    out = ["\\begin{tabular}{@{}c@{}}",
           "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l@{\\hspace{4pt}}l" + "@{\\hspace{4pt}}r" * 6 + "@{}}",
           "\\toprule",
           "\\multicolumn{8}{@{}l}{\\emph{Panel A: EM / F1 (\\%)}} \\\\",
           " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASET_LABELS[d]}}}" for d in DATASETS) + " \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
           "Model & Condition & EM & F1 & EM & F1 & EM & F1 \\\\"]
    for model in MODELS:
        out += ["\\midrule"] + panel_a_rows(summary, model)
    out += ["\\midrule", "\\end{tabular*} \\\\",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l@{\\hspace{3pt}}l" + "@{\\hspace{3pt}}c" * len(DATASETS) + "@{}}",
            "\\multicolumn{5}{@{}l}{\\emph{Panel B: EM effect (pp), estimate [paired bootstrap 95\\% CI]}} \\\\",
            "Model & Effect & " + " & ".join(DATASET_LABELS[d] for d in DATASETS) + " \\\\"]
    for model in MODELS:
        out += ["\\midrule"] + panel_b_rows(ci, model)
    return out + ["\\bottomrule", "\\end{tabular*}", "\\end{tabular}"]


def efficiency_rows(summary, model):
    out = []
    for i, cond in enumerate(CONDITIONS):
        cells = [model_label(model) if i == 0 else "", CONDITION_LABELS[cond]]
        for ds in DATASETS:
            r = summary[(ds, model, cond)]
            cells += [verbatim(r[k]) for k in EFFICIENCY]
        out.append(row(cells, f"model={model} condition={cond} datasets={','.join(DATASETS)}"))
    return out


def efficiency_panels(summary, style="grid"):
    n, ncols = len(EFFICIENCY), 2 + len(EFFICIENCY) * len(DATASETS)
    if style == "grid":
        heads = [f"\\multicolumn{{{n}}}{{c|}}{{Panel {letter}: {DATASET_LABELS[ds]}}}"
                 for letter, ds in zip("ABC", DATASETS)]
        out = [f"\\begin{{tabular}}{{{grid_spec('ll' + 'r' * (ncols - 2))}}}", "\\hline",
               "\\multicolumn{2}{|c|}{} & " + " & ".join(heads) + " \\\\", f"\\cline{{3-{ncols}}}",
               "Model & Condition & " + " & ".join(EFFICIENCY_HEADERS * len(DATASETS)) + " \\\\", "\\hline"]
        for model in MODELS:
            out += grid_group(efficiency_rows(summary, model), ncols)
        return out + ["\\end{tabular}"]
    spec = "@{}l@{\\hspace{3pt}}l" + ("@{\\hspace{8pt}}r" + "@{\\hspace{3pt}}r" * (n - 1)) * len(DATASETS) + "@{}"
    heads, rules = [], []
    for i, (letter, ds) in enumerate(zip("ABC", DATASETS)):
        heads.append(f"\\multicolumn{{{n}}}{{c}}{{Panel {letter}: {DATASET_LABELS[ds]}}}")
        first = 3 + i * n
        rules.append(f"\\cmidrule({'l' if i == len(DATASETS) - 1 else 'lr'}){{{first}-{first + n - 1}}}")
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
           " & & " + " & ".join(heads) + " \\\\", "".join(rules),
           "Model & Condition & " + " & ".join(EFFICIENCY_HEADERS * len(DATASETS)) + " \\\\"]
    for model in MODELS:
        out += ["\\midrule"] + efficiency_rows(summary, model)
    return out + ["\\bottomrule", "\\end{tabular}"]


def probe_rows(probe):
    def fmt(value, spec):
        return MISSING if value is None else format(float(value), spec)
    return [row([label] + [fmt(probe[m].get(key), spec) for m in MODELS],
                f"source=probe_results.json key={key} models={','.join(MODELS)}")
            for label, key, spec in (("Items", "n", ".0f"), ("Valid rate", "valid_tool_call_rate", ".2f"),
                                     ("Latency (ms)", "mean_latency_ms", ".0f"))]


def laya_groups(laya_rows, laya_overall):
    """Panel B rows, one group per question (FT, ZS) and the Overall group."""
    groups = []
    for q in LAYA_QUESTIONS:
        group = []
        for i, (model, label) in enumerate(LAYA_MODELS.items()):
            r = laya_rows[(q, model)]
            name = "\\texttt{" + q.replace("_", "\\_") + "}" if i == 0 else ""
            group.append(row([name, label] + [verbatim(r[k]) for k in ("accuracy", "Brier", "ECE", "MAE")],
                             f"source=laya_eval_report.md question={q} model={model}"))
        groups.append(group)
    group = []
    for i, (model, label) in enumerate(LAYA_MODELS.items()):
        acc, ece = laya_overall[model]
        group.append(row(["Overall" if i == 0 else "", label, acc, MISSING, ece, MISSING],
                         f"source=laya_eval_report.md overall model={model}"))
    return groups + [group]


def controller_probe(probe, laya_rows, laya_overall, style="grid"):
    groups = laya_groups(laya_rows, laya_overall)
    if style == "grid":
        out = [f"\\begin{{tabular}}{{{grid_spec('lrrrrr')}}}", "\\hline",
               "\\multicolumn{6}{|l|}{\\emph{Panel A: Function-calling probe}} \\\\", "\\hline",
               "Model & " + " & ".join(model_label(m) for m in MODELS) + " \\\\", "\\hline"]
        for line in probe_rows(probe):
            out += [line, "\\hline"]
        out += ["\\multicolumn{6}{|l|}{\\emph{Panel B: Decision layer on held-out decisions}} \\\\", "\\hline",
                "Question & & Acc. & Brier & ECE & MAE \\\\", "\\hline"]
        for group in groups:
            out += grid_group(group, 6)
        return out + ["\\end{tabular}"]
    out = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
           "\\multicolumn{6}{@{}l}{\\emph{Panel A: Function-calling probe}} \\\\",
           "Model & " + " & ".join(model_label(m) for m in MODELS) + " \\\\", "\\midrule"]
    out += probe_rows(probe)
    out += ["\\midrule",
            "\\multicolumn{6}{@{}l}{\\emph{Panel B: Decision layer on held-out decisions}} \\\\",
            "Question & & Acc. & Brier & ECE & MAE \\\\", "\\midrule"]
    out += [line for group in groups[:-1] for line in group] + ["\\midrule"] + groups[-1]
    return out + ["\\bottomrule", "\\end{tabular}"]


def write(lines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(PAPER.parent / "data" / "runs"))
    ap.add_argument("--run-id", default="main-20260930")
    ap.add_argument("--ci", default=str(PAPER / "analysis" / "bootstrap_ci.csv"))
    ap.add_argument("--tables", default=str(PAPER / "tables"))
    ap.add_argument("--style", choices=["grid", "booktabs"], default="grid",
                    help="grid: ruled cells as in the reference paper (R9, default); booktabs: the R8 tables")
    a = ap.parse_args(argv)
    run_dir = Path(a.runs) / a.run_id
    try:
        summary = load_summary(run_dir / "summary.csv")
        ci = load_ci(Path(a.ci))
        probe = load_probe(run_dir / "probe_results.json")
        laya_rows, laya_overall = load_laya(Path(a.runs) / "laya_eval" / "laya_eval_report.md")
    except (CheckError, OSError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        return 1
    tables = Path(a.tables)
    write(main_results_panels(summary, ci, a.style), tables / "main_results_panels.tex")
    write(efficiency_panels(summary, a.style), tables / "efficiency_panels.tex")
    write(controller_probe(probe, laya_rows, laya_overall, a.style), tables / "controller_probe.tex")
    write(effects_appendix(ci, a.style), tables / "effects_recall_retrieval.tex")      # R9 round 2: appendix B
    write(efficiency_p95(summary, a.style), tables / "efficiency_p95.tex")             # R9 round 2: appendix B
    print(f"OK: main_results_panels.tex, efficiency_panels.tex, controller_probe.tex, effects_recall_retrieval.tex, "
          f"efficiency_p95.tex written to {tables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
