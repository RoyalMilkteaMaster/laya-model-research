# Publication provenance and verification

This repository is a curated export of the original research project, prepared on
**2026-10-09**. It contains the multi-hop QA study, not the separate coffee-order
tutorial, personal application documents, machine configuration, or development chat history.

## Source and measurements

- Original source HEAD: `d86c38b` (full SHA and per-file hashes are in
  [source_snapshot.json](../provenance/source_snapshot.json)).
- Original benchmark: `main-20260930`.
- Run launch source commits: `fd11d729df600225dc0e124ae02095276846dac9` and
  `25569ed782f134e4cd37cc68ffa649fed9561432`. The later commit fixes manifest merging
  on resume; it does not change the question-answering algorithm.
- The exported source includes subsequent reporting/figure-label fixes. The agent loop,
  retrieval, controller training, and answer scoring retain the research implementation.
- The paper includes the author's existing later wording/layout edits from the
  source working tree. They were not rewritten as part of this publication.
- Full original per-question JSONL files, including failed attempts before retries, are
  copied byte-for-byte. Dataset JSONL and preparation manifests are also copied unchanged.
- Four run/training/evaluation metadata files replace machine-local paths with
  `${PROJECT_DATA_ROOT}`, `${PROJECT_RUNTIME_ROOT}`, or `${CODE_ROOT}` placeholders.
  The affected filenames and original hashes are recorded in `source_snapshot.json`.
  These placeholders describe the original location; they are not paths to execute.

Historical Git SHAs identify the local research repository. This public repository has
a new curated history; those SHAs are not commits that can be checked out here.
Use the preserved file hashes to compare an original copy, and this repository's release
tag/commit to identify the public artifact.

## Publication changes

Added bilingual introductions, result previews, dataset/license attribution, a CPU-only
reproduction command, publication checks, and GitHub Actions. The environment example
and analysis-script defaults now use portable paths. The setup script installs from
the existing lockfile with `--locked`. No model output or reported metric was edited.

`scripts/reproduce.py` writes new outputs outside the frozen data. The checksummed
inputs in `provenance/SHA256SUMS` include the paper run, dataset snapshot, controller
report, CSV analyses, and reference LaTeX tables. Downloaded license texts are retained
with links to their upstream sources.

## Verified during publication

1. CPU reproduction passed both in the existing research environment and a separate
   fresh Python 3.12 virtual environment installed only from `requirements-reproduce.txt`.
2. All 12,000 final records' EM/F1 values were recalculated from saved predictions and
   reference answers. Every cell's question IDs matched the frozen 200-question sample.
3. All 60 summary rows, 60 bootstrap effects, 280 taxonomy rows, and 13 LaTeX tables
   matched the archived values/text. Five figures were regenerated and previews inspected.
4. The original CPU/unit suite and analysis tests passed: **488 passed, 7 hardware
   integration tests deselected**. Four additional publication integrity tests cover
   edited answers, missing records, changed file hashes, and protected output paths.
5. `uv lock --check --project multihop_benchmark` passed. Exact versions and result
   verification are recorded in [results/verification.json](../results/verification.json).
6. The saved Laya checkpoint loaded on the GPU, its directory hash matched the
   original manifest, and its integration test passed. A separate 0.8B/HotpotQA smoke
   run evaluated two questions under all four conditions: 6 completed answers and
   2 invalid tool-call outputs. All 8 statuses and EM scores matched their original
   records. The two invalid outputs are preserved as failures, not hidden or retried.
   See [GPU smoke verification](../results/gpu_smoke_verification.json). This check
   reused the original local checkpoint/index; it did not validate retraining.

The CPU check does not remeasure latency or VRAM, rerun the complete 60-configuration
experiment, retrain Laya, or independently reconstruct the historical aggregate
controller-evaluation report. See [REPRODUCING.md](REPRODUCING.md) for those distinctions.

## Adding the repository to the paper

Suggested availability statement (the paper itself is unchanged):

> Code, frozen evaluation inputs, per-question outputs, and scripts for reproducing the
> reported tables and figures are available at
> https://github.com/RoyalMilkteaMaster/laya-multihop-qa.

Use an immutable release tag or commit when citing the repository in the paper.

## Version 1.0.1: paper narrative and presentation

The Chinese and English introductions now explain the progression from multi-hop
control failures to the generator/controller division, factorial experiment, weak
supervision, scale-dependent results, and error analysis. All current documentation
refers to the work as a paper. An additional effect plot presents the archived paired
confidence intervals across all three datasets. The underlying experiment code,
recorded outputs, reference tables, and paper PDF are unchanged.
