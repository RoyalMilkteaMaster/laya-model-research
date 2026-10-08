# Reproducing the study

There are two different tasks: recomputing the published statistics from saved outputs,
and producing new outputs by running the models. Only the first is expected to match exactly.

## 1. Recompute the published results on a CPU

Requires Python 3.12 and approximately 100 MB for the checked-out files, plus Python packages.
Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python scripts/reproduce.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
Alternatively invoke `.\.venv\Scripts\python.exe` directly for the last two commands.
No Ollama server, GPU, model weights, API key, or dataset download is required.

The script checks the frozen inputs against `provenance/SHA256SUMS`, matches every
configuration to its 200 question IDs, recomputes EM/F1 from the saved answers, and
rebuilds all 60 summary rows, 60 bootstrap effects, 280 error-taxonomy rows, 13 LaTeX
tables, and 5 figures. Output goes to `build/reproduced/`; the supplied data is read-only.
CSV cells and LaTeX text (with platform line endings normalized) must match the
archived outputs or the script exits nonzero. Figure PDF bytes are not compared across
library/platform versions; their underlying numeric inputs are verified.

`verification.json` records the checks and software versions. The historical controller
quality report is an input to Table 2: individual held-out controller predictions were
not archived, so this route does **not** independently recompute that report's accuracy,
Brier score, or ECE. Run `train-laya --eval-only` after training to evaluate the controller.

The README's additional cross-dataset effect overview is generated separately from the
same archived CSVs: `python scripts/plot_controller_effect.py`. It verifies all 15
RAG-controller effect estimates against `summary.csv` and writes PNG/PDF versions to
`results/figures/`. This is a presentation of existing results, not a sixth original paper figure.

## 2. Run the models again (Linux / WSL2, NVIDIA GPU)

The original machine used an RTX 4090 with 24 GB VRAM, Ubuntu on WSL2, Python 3.12.14,
Ollama 0.34.4, and Laya 0.3.21. The exact Python dependency resolution is in
`multihop_benchmark/uv.lock`. Five Qwen models plus the retrieval/controller stack
require tens of GB of model storage. The 27B configuration approached the GPU's
memory capacity; smaller GPUs may need a smaller model subset and cannot be assumed
to reproduce the reported latency or memory figures.

The original fine-tuned checkpoint is not distributed in this repository. Its recorded
hash, training parameters, calibration temperatures, and training/evaluation reports
are included. The following commands reconstruct it from the supplied training splits.
Model training and inference can vary with GPU kernels and upstream checkpoint changes;
seed 42 and temperature 0 do not guarantee identical weights or answers.

### Install and prepare a separate working data directory

From the repository root in Bash, with `curl` and `tar` available:

```bash
export PROJECT_DATA_ROOT="$PWD/.work/data"
export PROJECT_RUNTIME_ROOT="$HOME/laya-reproduction-runtime"
export OLLAMA_HOST=http://localhost:11434
bash multihop_benchmark/scripts/setup_wsl_env.sh
source multihop_benchmark/scripts/env.sh
cp -R data/datasets "$PROJECT_DATA_ROOT/"
```

The setup script uses `uv sync --locked`, installs Ollama in the runtime directory,
and checks its archive against the release SHA256 manifest. It does not pull models.
The checked-in `data/runs/main-20260930` is an immutable reference: do not run new
experiments into that directory or reuse that run ID for a new experiment.

Start `ollama serve` in a second terminal **after sourcing the same `env.sh`**.
If an Ollama server is already running, use that server only if its model storage and
version match your intended setup. In the first terminal:

```bash
for size in 0.8b 2b 4b 9b 27b; do ollama pull "qwen3.5:$size"; done
uv run --locked --project multihop_benchmark multihop-benchmark setup-check
```

Compare the pulled model digests with `data/runs/main-20260930/manifest.json`.
Tags and unpinned Hugging Face model repositories can change; matching names alone
does not establish an identical checkpoint. Original Qwen digests are preserved;
the original run did not record every Hugging Face model revision.

### Build retrieval indexes and train Laya

```bash
uv run --locked --project multihop_benchmark multihop-benchmark build-index --device cuda --report
uv run --locked --project multihop_benchmark multihop-benchmark make-laya-data --seed 42
uv run --locked --project multihop_benchmark multihop-benchmark train-laya --device cuda
```

Expected generated decision counts: 32,451 train / 3,996 calibration / 4,032 test.
The original decision JSONL hashes are in `provenance/source_snapshot.json`.
Defaults match the archived training summary: 2 epochs, micro-batch 8, accumulation 8,
group size 4, encoder LR 2.5e-5, head LR 1e-4, seed 42. Training uses RLCD and a held-out
temperature calibration; it is not a LoRA workflow. All three fitted temperatures in
the original run reached the search upper bound of 5.0.

The supplied `data/datasets` already contains the exact normalized splits and corpus
used in the paper. To investigate preprocessing from upstream datasets, use
`multihop-benchmark prepare-data` in a **different empty working data directory**,
then compare its JSONL hashes to the archived `prepare_manifest.json` files. The old
loader did not pin HF dataset revisions, which is why frozen inputs are included here.

### Smoke test, then the full 60 configurations

```bash
uv run --locked --project multihop_benchmark multihop-benchmark probe-models --run-id smoke-public --models qwen3.5:0.8b
uv run --locked --project multihop_benchmark multihop-benchmark run --run-id smoke-public --models qwen3.5:0.8b --datasets hotpotqa --limit 2

uv run --locked --project multihop_benchmark multihop-benchmark probe-models --run-id replication
uv run --locked --project multihop_benchmark multihop-benchmark run --run-id replication
uv run --locked --project multihop_benchmark multihop-benchmark aggregate --run-id replication --paper-dir build/replication
uv run --locked --project multihop_benchmark multihop-benchmark report --run-id replication
```

The runner is resumable: rerunning the same command skips completed (`done`) records
and retries failed records. This can alter final failure counts; record every retry.
Do not use a smoke run ID for the full run because the manifest retains first-launch
settings. For a long detached run, install `tmux`, then use
`bash multihop_benchmark/scripts/start_benchmark_tmux.sh replication`.

To compute the analysis tables for a new run:

```bash
uv run --locked --project multihop_benchmark python paper/analysis/bootstrap_ci.py --runs "$PROJECT_DATA_ROOT/runs" --run-id replication --out-dir build/replication/analysis --table build/replication/tables/bootstrap_ci.tex
uv run --locked --project multihop_benchmark python paper/analysis/error_taxonomy.py --runs "$PROJECT_DATA_ROOT/runs" --run-id replication --datasets "$PROJECT_DATA_ROOT/datasets" --csv build/replication/analysis/error_taxonomy.csv --table build/replication/tables/error_taxonomy.tex
uv run --locked --project multihop_benchmark python paper/analysis/merge_tables.py --runs "$PROJECT_DATA_ROOT/runs" --run-id replication --ci build/replication/analysis/bootstrap_ci.csv --tables build/replication/tables
```

The table merger expects the controller evaluation at `$PROJECT_DATA_ROOT/runs/laya_eval`,
written by `train-laya`. Do not compare new inference outputs to the frozen run using
`scripts/reproduce.py`; that script intentionally verifies only the archived paper run.

## Tests

From the repository root in the full environment:

```bash
source multihop_benchmark/scripts/env.sh
uv run --locked --project multihop_benchmark pytest -q multihop_benchmark/tests paper/analysis tests -m 'not integration'
```

Hardware-dependent tests are marked `integration` and require a separately configured
local Ollama/GPU runtime. The CPU reproduction script is also run by GitHub Actions.
See [PROVENANCE.md](PROVENANCE.md) for what was actually validated during publication.
