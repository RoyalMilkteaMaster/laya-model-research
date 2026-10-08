# multihop_benchmark

Laya 作為多跳問答 Agent 決策層的對比實驗（uv 專案，Python 3.12）。
公開版入口：[研究介紹](../README.md)／[完整重現指南](../docs/REPRODUCING.md)。
以下保留 CLI 操作說明；第一次使用請依完整重現指南設定獨立的資料與模型目錄。

## 安裝（WSL 使用者空間，不需 sudo，可重複執行）

```bash
bash multihop_benchmark/scripts/setup_wsl_env.sh
```

腳本依序：安裝 uv 到 `~/.local/bin` → `uv python install 3.12` → 產生 `multihop_benchmark/.env`
（若不存在；範本 `.env.example`）→ 建立 Data Root／Runtime Root 子目錄 → 在 Runtime Root
`environments/multihop_benchmark/` 建 venv 並 `uv sync` → 下載 Ollama 官方 tarball
（`ollama-linux-amd64.tar.zst`，預設 `OLLAMA_VERSION=v0.34.4`）解壓至 Runtime Root
`applications/ollama/`。已存在的安裝物不重下載。本腳本不拉取任何模型。

Ollama tarball 每次解壓前都必須通過 release `sha256sum.txt` 比對；取不到 `sha256sum.txt`、
未列出此檔或雜湊不符時一律非零退出、不解壓（不符時刪除 tarball，其餘情況保留供續傳）。

`.env` 欄位：`PROJECT_DATA_ROOT`、`PROJECT_RUNTIME_ROOT`、`OLLAMA_HOST`（預設
`http://localhost:11434`）。已匯出的同名環境變數優先於 `.env`。

## 執行任一 CLI 子命令的標準做法

uv 與 ollama 不會讀 `.env`，先在 shell 載入 `scripts/env.sh`（匯出 `OLLAMA_MODELS`、`HF_HOME`、
`OLLAMA_HOST`、`UV_PROJECT_ENVIRONMENT`，並把 `applications/ollama/bin` 加入 `PATH`），再用
console script `multihop-benchmark` 執行（於 Code Root 下）：

```bash
source multihop_benchmark/scripts/env.sh
uv run --project multihop_benchmark multihop-benchmark setup-check
uv run --project multihop_benchmark multihop-benchmark clean-logs --older-than 90
```

未 source `env.sh` 時，`uv run` 會在專案目錄另建 `.venv`，HF 快取也會落到 `~/.cache`；
`setup-check` 會以 `OLLAMA_MODELS`／`HF_HOME` 項目失敗提醒。

## 啟動 Ollama（tmux）

```bash
tmux new-session -d -s laya-ollama "bash -c 'source $PWD/multihop_benchmark/scripts/env.sh && exec ollama serve'"
source multihop_benchmark/scripts/env.sh
curl -s -o /dev/null -w '%{http_code}\n' "$OLLAMA_HOST/api/tags"   # 預期 200
```

## 子命令

| 子命令 | 說明 |
|---|---|
| `setup-check` | 回報 GPU 名稱與 VRAM、Ollama 連線、Code／Data／Runtime Root 路徑、Ollama 執行檔與 `OLLAMA_MODELS`／`HF_HOME`；任一失敗非零退出並列出缺少項目。 |
| `clean-logs --older-than N` | 只刪除 Data Root `logs/` 下日期超過 N 天的輪替檔（`<name>-YYYY-MM-DD.jsonl`），不刪目前寫入中的檔案。 |
| `prepare-data [--datasets NAME ...] [--force]` | 經 `HF_HOME` 快取下載 `hotpotqa`／`2wiki`／`musique`（失敗重試、已快取不重下載），轉統一 schema，以 seed 42 抽 validation 200 題與 train 800／100／100 題，建 `corpus.jsonl`，寫到 Data Root `datasets/<dataset>/` 與 `datasets/LICENSES.md` 並印出行數；輸出已齊全時略過，`--force` 重建。 |
| `make-laya-data [--seed 42] [--audit-size 100]` | 由三集 `train_800／calibration_100／test_100` 以狀態模擬產生 Laya 弱標籤決策，寫 Data Root `laya_decisions/{train,calibration,test}.jsonl`，印出各狀態／型別筆數、state token 最大值（≤ 900）與標籤抽驗結果；抽驗不符時非零退出。 |
| `train-laya [--epochs N] [--limit N] [--skip-train \| --eval-only] [--device cuda]` | 以 `laya_decisions/train.jsonl` RLCD 微調 `convaiinnovations/laya-multilingual`、`calibration.jsonl` 校準溫度，輸出 Runtime Root `models/laya_multihop/`；再在 `test.jsonl` 評估微調模型與 zero-shot 基底，寫 Data Root `runs/laya_eval/{train_summary.json,laya_eval_report.md,laya_eval_report.json}`。未達門檻時報告標 FAIL 並非零退出。其他超參數旗標見 `--help`；`--limit` 只供冒煙。需 `TORCH_DISABLE_NATIVE_JIT=1`（CLI 會自動設定）。 |
| `probe-models --run-id X [--models TAG ...]` | 每個模型（預設五個 Qwen 3.5 tag）跑固定 20 題工具呼叫探針（要求呼叫 `search(query)`），寫 Data Root `runs/<run_id>/probe_results.json`（每模型 `valid_tool_call_rate`、`n`、`mean_latency_ms` 與逐題結果；已存在時以模型為單位合併），Log `probe_started`／`probe_finished`。請在 `run` 之前用同一個 `run_id` 執行，manifest 才會帶探針摘要。 |
| `run --run-id X [--limit N] [--models ...] [--datasets ...] [--conditions ...] [--timeout-s 300]` | 依模型分組（models → datasets → conditions，同模型各組連續、換模型前卸載）執行每題多跳迴圈，寫 `runs/<run_id>/records/<dataset>__<model slug>__<condition>.jsonl`（每題一行，含 `vram_peak_mb`、`finished_at`）與啟動時的 `manifest.json`。同 `run_id` 重啟即續跑：跳過 `status=done` 的題，其餘重跑並 append。OOM 以 `num_ctx=4096` 重試一次；逾時等單題錯誤只記錄不中止；Ollama 連線重試用盡時記 ERROR 並以退出碼 1 中止。 |

新增子命令：在 `multihop_benchmark/cli.py` 末端「子命令區塊」附加一個
`@subcommand(...)` 區塊即可，不需改動其他區塊（格式見該檔開頭說明）。

`report --run-id X [--every 600]`：單次或每 N 秒重寫 Data Root `reports/PROGRESS.md` 與自含式
`reports/dashboard.html`（瀏覽器以 `file://` 直接開啟）。report 只讀 `runs/<run_id>/records/*.jsonl`、
`runs/<run_id>/manifest.json` 與 `datasets/<dataset>/sample_200.jsonl`（僅補「最近 3 題」的題目文字），只寫
`reports/`。完成率、組別狀態與 ETA 只計 `status=done`；「最近錯誤」「最近 3 題」依 record 選填欄位
`finished_at` 排序，缺此欄位時以檔案 mtime 推定並標示順序為近似。

## 正式實驗（tmux 背景執行）

```bash
source multihop_benchmark/scripts/env.sh
uv run --project multihop_benchmark multihop-benchmark probe-models --run-id main
bash multihop_benchmark/scripts/start_benchmark_tmux.sh main            # 可附 run 參數，例 --limit 3
tmux attach -t laya-bench-main                                           # Ctrl-b d 離開，程式繼續執行
```

`start_benchmark_tmux.sh <run_id> [run 參數]` 建立 detached session `laya-bench-<run_id>`（可用
`LAYA_TMUX_SESSION` 覆寫；session 已存在時拒絕）三個視窗：`ollama`（Ollama 未回應時啟動 `ollama serve`，
已在執行時沿用並每 60 秒顯示 `ollama ps`）、`run`（等 Ollama 就緒後 `run --run-id <run_id>`）、`report`
（`report --run-id <run_id> --every 600`）。程式結束後視窗保留 shell 與退出碼。session 由 tmux server 持有，
關閉 MobaXterm／終端不影響；中斷後重新執行腳本（先 `tmux kill-session -t laya-bench-<run_id>`）即續跑。

## 正式 Log

各模組只呼叫 `benchmark_logger.get_logger(source)`：

```python
from multihop_benchmark.benchmark_logger import get_logger

logger = get_logger("run")
logger.info("開始組合", event="combo_started", run_id=run_id, combo=combo)
```

- 寫入 Data Root `logs/`，JSONL 每行含 `timestamp / level / event / source / message`，其他關鍵字參數
  （`run_id`、`combo`、`question_id`、`errorCode` 等）附為欄位；與保留欄位（`timestamp`、`source`、
  `message`、`exception`、`context`）撞名的參數不覆寫核心欄位，改放在巢狀 `context` 欄位。
- 檔名由 `source` 決定：`run`、`report` 與未列出的 source → `benchmark.jsonl`；`train-laya` →
  `train_laya.jsonl`；`prepare-data`、`build-index`、`make-laya-data` → `prepare_data.jsonl`。
  程式庫模組可用 `get_logger("dense_index", log_name="prepare_data")` 指定檔案。
- 每日午夜輪替，舊檔改名 `<name>-YYYY-MM-DD.jsonl`；不自動刪除，保存 90 天後以 `clean-logs` 手動清理。

## 測試

```bash
source multihop_benchmark/scripts/env.sh
uv run --project multihop_benchmark pytest multihop_benchmark      # 單元測試（預設排除 integration）
uv run --project multihop_benchmark pytest multihop_benchmark -m integration   # 需 Ollama＋GPU
```
