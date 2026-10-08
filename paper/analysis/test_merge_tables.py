"""Tests for merge_tables.py on synthetic inputs (no model, no Data Root).

Run:  source multihop_benchmark/scripts/env.sh && uv run --project multihop_benchmark pytest paper/analysis
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("merge_tables.py")
DS = ["hotpotqa", "2wiki", "musique"]
MODELS = ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b"]
CONDS = ["recall_llm", "rag_llm", "recall_laya", "rag_laya"]
FIELDS = ["dataset", "model", "condition", "n", "em", "f1", "latency_p50_ms", "latency_p95_ms", "vram_peak_mb",
          "prompt_tokens_mean", "output_tokens_mean", "hops_mean", "laya_latency_ms_mean"]
LAYA_MD = """# report

| 問題 | 模型 | n | accuracy | soft accuracy | Brier | ECE | MAE | MAE (argmax) | within-one |
|---|---|---|---|---|---|---|---|---|---|
| sufficient | 微調 | 9 | 0.9001 | 0.8 | 0.1002 | 0.0603 | — | — | — |
| sufficient | zero-shot | 9 | 0.5004 | 0.5 | 0.5005 | 0.2006 | — | — | — |
| next_action | 微調 | 9 | 0.8007 | 0.8 | 0.3008 | 0.1309 | — | — | — |
| next_action | zero-shot | 9 | 0.3010 | 0.3 | 0.7011 | 0.2112 | — | — | — |
| remaining_hops | 微調 | 9 | 0.8113 | — | — | 0.1414 | 0.2015 | 0.1 | 0.9 |
| remaining_hops | zero-shot | 9 | 0.3016 | — | — | 0.3717 | 0.8718 | 0.9 | 0.6 |

- 整體 accuracy：微調 0.8519、zero-shot 0.4120；整體 ECE：微調 0.1121、zero-shot 0.2722
"""


def make_inputs(tmp_path, drop=None):
    run = tmp_path / "runs" / "main-20260930"
    run.mkdir(parents=True)
    with open(run / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for d, ds in enumerate(DS):
            for m, model in enumerate(MODELS):
                for c, cond in enumerate(CONDS):
                    if drop == (ds, model, cond):
                        continue
                    k = f"{d}{m}{c}"   # unique digits per cell
                    w.writerow(dict(dataset=ds, model=model, condition=cond, n="200", em=f"1{k}.5", f1=f"2{k}.5",
                                    latency_p50_ms=f"3{k}", latency_p95_ms=f"9{k}", vram_peak_mb=f"4{k}",
                                    prompt_tokens_mean=f"5{k}", output_tokens_mean=f"6{k}", hops_mean=f"7.{k}",
                                    laya_latency_ms_mean=(f"8{k}.5" if "laya" in cond else "")))
    (run / "probe_results.json").write_text(json.dumps({"models": {
        m: {"n": 20, "valid_tool_call_rate": 1.0 - i / 100, "mean_latency_ms": 100.4 + i} for i, m in enumerate(MODELS)}}),
        encoding="utf-8")
    laya = tmp_path / "runs" / "laya_eval"
    laya.mkdir()
    (laya / "laya_eval_report.md").write_text(LAYA_MD, encoding="utf-8")
    ci = tmp_path / "bootstrap_ci.csv"
    with open(ci, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "effect", "estimate", "ci_low", "ci_high", "n", "B", "seed"])
        for d, ds in enumerate(DS):
            for m, model in enumerate(MODELS):
                for e, eff in enumerate(["controller_rag", "controller_recall", "retrieval_llm", "interaction"]):
                    w.writerow([ds, model, eff, f"-{d + 1}{m}{e}.5", f"-9{d}{m}{e}.0", f"{d + 1}{m}{e}.0", 200, 10000, 1])
    return tmp_path / "runs", ci


def run_script(runs, ci, out):
    return subprocess.run([sys.executable, str(SCRIPT), "--runs", str(runs), "--ci", str(ci), "--tables", str(out)],
                          capture_output=True, text=True)


def row(tex, *keys):
    return next(line for line in tex.splitlines() if all(k in line for k in keys))


def cells(line):
    return [c.strip() for c in line.split("%")[0].rstrip().rstrip("\\").split("&")]


def test_main_results_panels_copy_cells_verbatim(tmp_path):
    runs, ci = make_inputs(tmp_path)
    p = run_script(runs, ci, tmp_path / "t")
    assert p.returncode == 0, p.stdout + p.stderr
    tex = (tmp_path / "t" / "main_results_panels.tex").read_text(encoding="utf-8")
    assert "Panel A" in tex and "Panel B" in tex
    # Panel A: model 9B (m=1), rag_laya (c=3): EM / F1 for hotpotqa (d=0), 2wiki (d=1), musique (d=2)
    a = row(tex, "model=qwen3.5:9b condition=rag_laya")
    assert cells(a)[2:] == ["1013.5", "2013.5", "1113.5", "2113.5", "1213.5", "2213.5"]
    # Panel B (R8 booktabs layout, kept as --style booktabs): interaction (e=3) of 0.8B (m=4) in the three datasets
    p = subprocess.run([sys.executable, str(SCRIPT), "--runs", str(runs), "--ci", str(ci), "--tables", str(tmp_path / "bt"),
                        "--style", "booktabs"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    tex = (tmp_path / "bt" / "main_results_panels.tex").read_text(encoding="utf-8")
    b = row(tex, "model=qwen3.5:0.8b effect=interaction")
    assert [re.findall(r"-?\d+\.\d", c) for c in cells(b)[2:]] == [
        ["-143.5", "-9043.0", "143.0"], ["-243.5", "-9143.0", "243.0"], ["-343.5", "-9243.0", "343.0"]]
    # Panel B is its own tabular (one cell per dataset), nested with Panel A in one outer tabular,
    # so its wide CI cells do not widen the EM / F1 columns of Panel A.
    assert len(cells(b)) == 5 and "multicolumn" not in b
    # Panel B lists the three effects of every model: under RAG, under recall, and the interaction
    rec = row(tex, "model=qwen3.5:0.8b effect=controller_recall")
    assert [re.findall(r"-?\d+\.\d", c) for c in cells(rec)[2:]][0] == ["-141.5", "-9041.0", "141.0"]
    effects = [line.split("effect=")[1].split()[0] for line in tex.splitlines() if "effect=" in line]
    assert effects == ["controller_rag", "controller_recall", "interaction"] * 5
    assert tex.count("\\begin{tabular") == 3 and tex.count("\\end{tabular") == 3


def test_efficiency_panels_side_by_side(tmp_path):
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "t").returncode == 0
    tex = (tmp_path / "t" / "efficiency_panels.tex").read_text(encoding="utf-8")
    assert all(f"Panel {x}" in tex for x in "ABC")
    r = row(tex, "model=qwen3.5:2b condition=rag_llm")      # m=3, c=1: no Laya latency
    per_ds = lambda d: [f"3{d}31", f"4{d}31", f"5{d}31", f"6{d}31", f"7.{d}31", "---"]
    assert cells(r)[2:] == per_ds(0) + per_ds(1) + per_ds(2)
    r = row(tex, "model=qwen3.5:2b condition=rag_laya")
    assert cells(r)[-1] == "8233.5"


def test_controller_probe_panels(tmp_path):
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "t").returncode == 0
    tex = (tmp_path / "t" / "controller_probe.tex").read_text(encoding="utf-8")
    assert cells(row(tex, "source=probe_results.json key=valid_tool_call_rate"))[1:] == [
        "1.00", "0.99", "0.98", "0.97", "0.96"]
    assert cells(row(tex, "source=probe_results.json key=mean_latency_ms"))[1:] == ["100", "101", "102", "103", "104"]
    assert cells(row(tex, "question=remaining_hops model=微調"))[1:] == ["FT", "0.8113", "---", "0.1414", "0.2015"]
    assert cells(row(tex, "overall model=zero-shot"))[1:] == ["ZS", "0.4120", "---", "0.2722", "---"]


def test_byte_identical_bare_tabulars_and_missing_input_fails(tmp_path):
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "a").returncode == 0
    assert run_script(runs, ci, tmp_path / "b").returncode == 0
    for name in ("main_results_panels.tex", "efficiency_panels.tex", "controller_probe.tex", "efficiency_p95.tex",
                 "effects_recall_retrieval.tex"):
        a = (tmp_path / "a" / name).read_bytes()
        assert a == (tmp_path / "b" / name).read_bytes()
        text = a.decode("utf-8")
        assert text.startswith("\\begin{tabular}") and text.rstrip().endswith("\\end{tabular}")
        assert "\\caption" not in text and "\\label" not in text
    runs2, ci2 = make_inputs(tmp_path / "x", drop=("2wiki", "qwen3.5:4b", "rag_llm"))
    p = run_script(runs2, ci2, tmp_path / "c")
    assert p.returncode == 1 and "2wiki/qwen3.5:4b/rag_llm" in p.stdout


def column_specs(tex):
    """Column specs of every tabular / tabular* in tex (balanced braces; tabular* skips its width)."""
    specs = []
    for m in re.finditer(r"\\begin\{tabular(\*?)\}", tex):
        i, groups = m.end(), []
        for _ in range(2 if m.group(1) else 1):
            depth, j = 0, i
            while True:
                depth += {"{": 1, "}": -1}.get(tex[j], 0)
                j += 1
                if depth == 0:
                    break
            groups.append(tex[i + 1:j - 1])
            i = j
        specs.append(groups[-1])
    return specs


def strip_at(spec):
    """Remove @{...} and >{...} expressions (balanced braces) from a column spec; p{width} becomes p."""
    out, i = "", 0
    while i < len(spec):
        if spec.startswith("p{", i):
            out += "p"
            depth, i = 0, i + 1
            while True:
                depth += {"{": 1, "}": -1}.get(spec[i], 0)
                i += 1
                if depth == 0:
                    break
        elif spec.startswith("@{", i) or spec.startswith(">{", i):
            depth, i = 0, i + 1
            while True:
                depth += {"{": 1, "}": -1}.get(spec[i], 0)
                i += 1
                if depth == 0:
                    break
        else:
            out, i = out + spec[i], i + 1
    return out


def assert_full_grid(tex):
    """Revision R9 (reference paper, IEEE AVSS 2025): every column is ruled and every row is followed by a
    horizontal rule; no booktabs rules. A one-column @{}c@{} wrapper (stacked panels) is exempt."""
    for bad in ("\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule"):
        assert bad not in tex, bad
    ruled = [strip_at(s) for s in column_specs(tex) if strip_at(s) != "c"]
    assert ruled and all(re.fullmatch(r"\|(?:[lrcp]\|)+", s) for s in ruled), ruled
    lines = tex.splitlines()
    rows = [i for i, line in enumerate(lines) if "\\\\" in line.split("%")[0] and not line.startswith("\\end")]
    assert rows and all(re.match(r"\\(hline|cline)", lines[i + 1]) for i in rows), \
        [lines[i:i + 2] for i in rows if not re.match(r"\\(hline|cline)", lines[i + 1])]
    ends = [i for i, line in enumerate(lines) if line.startswith("\\end{tabular") and not lines[i - 1].startswith("\\end")]
    assert ends and all(lines[i - 1] == "\\hline" for i in ends)


def test_default_style_is_full_grid_and_booktabs_is_kept_as_an_option(tmp_path):
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "g").returncode == 0
    p = subprocess.run([sys.executable, str(SCRIPT), "--runs", str(runs), "--ci", str(ci),
                        "--tables", str(tmp_path / "b"), "--style", "booktabs"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    for name in ("main_results_panels.tex", "efficiency_panels.tex", "controller_probe.tex", "efficiency_p95.tex",
                 "effects_recall_retrieval.tex"):
        grid = (tmp_path / "g" / name).read_text(encoding="utf-8")
        booktabs = (tmp_path / "b" / name).read_text(encoding="utf-8")
        assert_full_grid(grid)
        assert "\\toprule" in booktabs and "|" not in booktabs.replace("\\\\", "")
        # same data rows (cells and source comments) in both styles; R9 round 2: the grid Panel B of
        # main_results_panels has one row per model and dataset, so only its Panel A rows are compared
        data = lambda t: [cells(x) + [x.split("%", 1)[1]] for x in t.splitlines()
                          if "% " in x and "&" in x and not (name == "main_results_panels.tex" and "effect" in x)]
        assert data(grid) == data(booktabs) and data(grid)


def test_grid_panel_b_has_one_row_per_model_and_dataset_with_rag_and_interaction_effects(tmp_path):
    """R9 round 2: at 7 pt the grid Panel B fits the column with the datasets as rows and two effect columns;
    the recall effect moves to the appendix table effects_recall_retrieval.tex."""
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "t").returncode == 0
    tex = (tmp_path / "t" / "main_results_panels.tex").read_text(encoding="utf-8")
    keys = [line.split("% ")[1] for line in tex.splitlines() if "effects=" in line]
    assert keys == [f"model={m} dataset={d} effects=controller_rag,interaction" for m in MODELS for d in DS]
    # 0.8B (m=4) on MuSiQue (d=2): controller_rag (e=0) and interaction (e=3), estimate [low, high] verbatim
    r = cells(row(tex, "model=qwen3.5:0.8b dataset=musique"))
    assert r[:2] == ["", "MuSiQue"]
    assert [re.findall(r"-?\d+\.\d", c) for c in r[2:]] == [["-340.5", "-9240.0", "340.0"], ["-343.5", "-9243.0", "343.0"]]
    assert cells(row(tex, "model=qwen3.5:0.8b dataset=hotpotqa"))[:2] == ["0.8B", "HotpotQA"]
    assert "Delta_{\\mathrm{RAG}}" in tex and "Delta_{\\mathrm{int}}" in tex and "Delta_{\\mathrm{Rec}}" not in tex


def test_appendix_tables_recall_and_retrieval_effects_and_p95_latency(tmp_path):
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "t").returncode == 0
    tex = (tmp_path / "t" / "effects_recall_retrieval.tex").read_text(encoding="utf-8")
    # 9B (m=1) on 2Wiki (d=1): controller_recall (e=1) and retrieval_llm (e=2)
    r = cells(row(tex, "model=qwen3.5:9b dataset=2wiki effects=controller_recall,retrieval_llm"))
    assert [re.findall(r"-?\d+\.\d", c) for c in r[2:]] == [["-211.5", "-9111.0", "211.0"], ["-212.5", "-9112.0", "212.0"]]
    assert len([x for x in tex.splitlines() if "effects=" in x]) == len(MODELS) * len(DS)
    p95 = (tmp_path / "t" / "efficiency_p95.tex").read_text(encoding="utf-8")
    # 2B (m=3), rag_llm (c=1): latency_p95_ms of the three datasets, verbatim
    assert cells(row(p95, "model=qwen3.5:2b condition=rag_llm"))[2:] == ["9031", "9131", "9231"]
    assert len([x for x in p95.splitlines() if "condition=" in x]) == len(MODELS) * len(CONDS)
    assert_full_grid(tex)
    assert_full_grid(p95)


def assert_rows_match_columns(tex):
    """R9 round 3 (Reviewer B, B-1): in every ruled tabular, each row spans exactly the columns of its spec
    (a \\multicolumn{n} counts n), every \\multicolumn closes with a rule and opens with one in the first column,
    and no \\extracolsep glue is put next to the rules (it shifted the rules of Table 3 off the \\cline ends)."""
    assert "\\extracolsep" not in tex
    ncols, checked = None, 0
    for line in tex.splitlines():
        if line.startswith("\\begin{tabular"):
            spec = column_specs(line)[0]
            ncols = len(strip_at(spec).replace("|", ""))
            # the cell padding is \\tabcolsep (set by the hook macros of main.tex), never an @{} next to a rule:
            # with |@{...} the padding lies outside the cell and every \\cline starts off its rule
            assert ncols == 1 or "@{" not in spec, spec
            continue
        body = line.split("%")[0]
        if "\\\\" not in body or line.startswith("\\end") or ncols in (None, 1):
            continue
        n = 0
        for j, cell in enumerate(body.rstrip().rstrip("\\").split("&")):
            m = re.search(r"\\multicolumn\{(\d+)\}\{([^}]*)\}", cell)
            if m:
                n += int(m.group(1))
                assert m.group(2).endswith("|") and (j > 0 or m.group(2).startswith("|")), line
            else:
                n += 1
        assert n == ncols, (n, ncols, line)
        checked += 1
    assert checked


def test_grid_rows_span_every_column_with_consistent_rules(tmp_path):
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "t").returncode == 0
    for name in ("main_results_panels.tex", "efficiency_panels.tex", "controller_probe.tex", "efficiency_p95.tex",
                 "effects_recall_retrieval.tex"):
        assert_rows_match_columns((tmp_path / "t" / name).read_text(encoding="utf-8"))


def test_grid_panels_of_main_results_have_fixed_columns_that_fill_the_column_width(tmp_path):
    """Both panels of Table 3 are plain tabulars whose fixed column widths add up to \\columnwidth, with the
    same Model and label column widths, so that the two panels are equally wide and their rules meet."""
    runs, ci = make_inputs(tmp_path)
    assert run_script(runs, ci, tmp_path / "t").returncode == 0
    tex = (tmp_path / "t" / "main_results_panels.tex").read_text(encoding="utf-8")
    specs = [s for s in column_specs(tex) if strip_at(s) != "c"]
    assert len(specs) == 2 and "tabular*" not in tex
    first_two = [re.findall(r"p\{([^{}]*)\}", s)[:2] for s in specs]
    assert first_two[0] == first_two[1] and all(w.endswith("pt") for w in first_two[0])
    assert all("\\columnwidth" in s for s in specs)
