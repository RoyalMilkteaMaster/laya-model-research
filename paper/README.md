# Paper and analysis

**Calibrated Loop Control for Multi-Hop QA with Small Language Models**,
by Pin-Hung Lin and Te-Lun Yang. [PDF](main.pdf) · [Traditional Chinese](zh-TW/main.md).

## Regenerate the results

From the repository root, follow [the CPU reproduction instructions](../docs/REPRODUCING.md)
and run `python scripts/reproduce.py`. It verifies the frozen data and writes new tables,
figures, and a verification report to `build/reproduced/` without changing this paper.
[Results mapping](../results/README.md) identifies the input data and script for each table.

`analysis/bootstrap_ci.py`, `error_taxonomy.py`, and `merge_tables.py` are the original
analysis scripts. Their default inputs now resolve to the included `data/` directory.
Use explicit output arguments, or the wrapper above, to preserve this frozen paper snapshot.

## Build the paper

The included PDF is the saved paper output. To compile the multi-file LaTeX
project with an existing Tectonic installation, from the repository root:

```bash
tectonic paper/main.tex
```

Alternatively upload the contents of `paper/` to Overleaf, set `main.tex` as the main
document, and compile with pdfLaTeX. The `sections/`, `tables/`, `figures/`, bibliography,
and style files must all be present. For the historical layout checker:

```bash
bash paper/check_paper.sh --compile
```

It expects the original eight-page main text/reference layout followed by appendices.
The checker requires a Bash/Linux environment and Tectonic; it is separate from CPU
result reproduction. The Chinese `build_pdf.py` helper requires its original document
rendering dependencies and is not part of the supported quick-start workflow.

## Content

- `sections/`: English paper sections.
- `tables/`, `figures/`: archived generated tables and vector figures.
- `analysis/`: bootstrap intervals, error taxonomy, panel-table generation and tests.
- `refs.bib`: citations to datasets, methods, models and software.
- `zh-TW/main.md`: paragraph-aligned Chinese translation.

The paper and third-party formatting files are excluded from the repository's
MIT code license. See [third-party notices](../THIRD_PARTY_NOTICES.md).
