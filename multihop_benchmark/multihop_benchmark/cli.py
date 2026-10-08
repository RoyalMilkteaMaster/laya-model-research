"""multihop-benchmark 命令列入口。

新增子命令的方式：在檔案末端「子命令區塊」附加自己的區塊，不改動其他區塊：

    # ---- <name> ----
    def _add_<name>_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(...)

    @subcommand("<name>", help="...", add_arguments=_add_<name>_arguments)
    def <name>_command(args: argparse.Namespace) -> int:
        from multihop_benchmark.xxx import yyy   # 重依賴在函式內匯入，保持 CLI 啟動快速
        ...
        return 0

處理函式回傳退出碼（0 成功，非零失敗）。
"""

from __future__ import annotations

import os

# Laya（ModernBERT）在 torch 2.14 需關閉 native JIT，且必須在任何子命令第一次匯入 torch 前設定（Spec 補充）。
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import argparse
import json
import logging
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from multihop_benchmark.benchmark_logger import delete_rotated_logs, get_logger
from multihop_benchmark.settings import SettingsError, load_settings

Handler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True)
class _Subcommand:
    name: str
    help: str
    handler: Handler
    add_arguments: Callable[[argparse.ArgumentParser], None] | None


_SUBCOMMANDS: dict[str, _Subcommand] = {}


def subcommand(
    name: str, *, help: str, add_arguments: Callable[[argparse.ArgumentParser], None] | None = None
) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        if name in _SUBCOMMANDS:
            raise ValueError(f"子命令重複註冊：{name}")
        _SUBCOMMANDS[name] = _Subcommand(name, help, handler, add_arguments)
        return handler

    return register


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multihop-benchmark", description="Laya 多跳問答對比實驗")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for command in _SUBCOMMANDS.values():
        subparser = subparsers.add_parser(command.name, help=command.help, description=command.help)
        if command.add_arguments:
            command.add_arguments(subparser)
        subparser.set_defaults(handler=command.handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


# ============================ 子命令區塊 ============================


# ---- setup-check ----
def _check(ok: bool, label: str, detail: str, missing: list[str]) -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {label}：{detail}")
    if not ok:
        missing.append(label)


def _query_gpu() -> tuple[bool, str]:
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"無法執行 nvidia-smi（{error}）"
    gpus = [line.rsplit(",", 1) for line in result.stdout.strip().splitlines() if line.strip()]
    if not gpus:
        return False, "nvidia-smi 沒有回報任何 GPU"
    return True, "; ".join(f"{name.strip()}, {memory.strip()} MiB" for name, memory in gpus)


def _probe_ollama(host: str) -> tuple[bool, str]:
    url = f"{host}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = json.load(response).get("models", [])
            return True, f"{url} 回 {response.status}，已安裝模型 {len(models)} 個"
    except (urllib.error.URLError, OSError, ValueError) as error:
        return False, f"無法連線 {url}（{error}）；請在 tmux 啟動 `ollama serve`"


@subcommand("setup-check", help="檢查 GPU、Ollama 連線與三根目錄")
def setup_check_command(args: argparse.Namespace) -> int:
    try:
        settings = load_settings()
    except SettingsError as error:
        print(f"[FAIL] 設定：{error}")
        return 2

    missing: list[str] = []
    print(f"Code Root：{settings.code_root}")
    gpu_ok, gpu_detail = _query_gpu()
    _check(gpu_ok, "GPU", gpu_detail, missing)
    ollama_ok, ollama_detail = _probe_ollama(settings.ollama_host)
    _check(ollama_ok, "Ollama", ollama_detail, missing)
    setup_hint = "（不存在，請執行 scripts/setup_wsl_env.sh）"
    for label, path in (
        ("Data Root", settings.data_root),
        ("Runtime Root", settings.runtime_root),
        ("Ollama 執行檔", settings.ollama_bin),
    ):
        _check(path.exists(), label, f"{path}{'' if path.exists() else setup_hint}", missing)
    for variable, expected in (("OLLAMA_MODELS", settings.ollama_models_dir), ("HF_HOME", settings.hf_home)):
        actual = os.environ.get(variable)
        ok = actual == str(expected)
        hint = "" if ok else f"（應為 {expected}，請 source scripts/env.sh）"
        _check(ok, variable, f"{actual or '未設定'}{hint}", missing)

    if settings.data_root.is_dir():
        get_logger("setup-check").log(
            logging.ERROR if missing else logging.INFO,
            "setup-check 完成", event="setup_check_finished", missing=missing,
        )
    if missing:
        print(f"缺少：{', '.join(missing)}")
        return 1
    print("全部檢查通過")
    return 0


# ---- clean-logs ----
def _non_negative_int(value: str) -> int:
    days = int(value)
    if days < 0:
        raise argparse.ArgumentTypeError("必須是 0 以上的整數")
    return days


def _add_clean_logs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--older-than", type=_non_negative_int, required=True, metavar="N", help="刪除日期超過 N 天的輪替檔"
    )


@subcommand("clean-logs", help="刪除超過 N 天的輪替 Log 檔（不刪目前寫入中的檔案）", add_arguments=_add_clean_logs_arguments)
def clean_logs_command(args: argparse.Namespace) -> int:
    logs_dir = load_settings().logs_dir
    if not logs_dir.is_dir():
        print(f"Log 目錄不存在，無需清理：{logs_dir}")
        return 0
    deleted = delete_rotated_logs(logs_dir, args.older_than)
    for path in deleted:
        print(f"已刪除 {path}")
    print(f"共刪除 {len(deleted)} 個超過 {args.older_than} 天的輪替檔（{logs_dir}）")
    get_logger("clean-logs").info(
        f"刪除 {len(deleted)} 個輪替檔", event="logs_cleaned",
        older_than_days=args.older_than, deleted=[path.name for path in deleted],
    )
    return 0


# ---- report ----
def _report_positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必須是正整數")
    return number


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True, help="Data Root runs/<run_id>/")
    parser.add_argument(
        "--every", type=_report_positive_int, metavar="SECONDS", help="每 N 秒重寫一次（tmux 常駐用 600）；省略則只寫一次"
    )
    parser.add_argument(
        "--count", type=_report_positive_int, metavar="N", help="搭配 --every：重寫 N 次後結束（預設持續到中斷）"
    )


def _write_reports_once(data_root, run_id: str) -> None:
    from multihop_benchmark.reporting.dashboard_builder import write_dashboard
    from multihop_benchmark.reporting.progress_report_writer import (
        NO_DATA, collect_progress, format_timestamp, percent, write_progress_report,
    )

    snapshot = collect_progress(data_root, run_id)
    progress_path = write_progress_report(data_root, run_id, snapshot=snapshot)
    dashboard_path = write_dashboard(data_root, run_id, snapshot=snapshot)
    stamp = format_timestamp(snapshot.generated_at)
    if not snapshot.has_data:
        summary = f"{NO_DATA}（{snapshot.records_dir}）"
    else:
        current = snapshot.current_combo.combo if snapshot.current_combo else "無"
        summary = (
            f"完成 {snapshot.done} / {snapshot.target_total} 題"
            f"（{percent(snapshot.done, snapshot.target_total)}），已嘗試 {snapshot.attempted}，目前組別 {current}"
        )
    print(f"[{stamp}] {summary}\n  → {progress_path}\n  → {dashboard_path}", flush=True)
    logger = get_logger("report")
    invalid_by_file: dict[str, list[int]] = {}
    for name, number in snapshot.invalid_lines:
        invalid_by_file.setdefault(name, []).append(number)
    for name, numbers in invalid_by_file.items():  # 每輪每檔最多一筆
        logger.warning(
            f"略過 {len(numbers)} 行無法解析的 record：{name}", event="record_line_invalid", run_id=run_id,
            errorCode="RECORD_LINE_INVALID", file=name, lines=numbers,
        )
    logger.info(
        f"報告已更新：{summary}", event="report_written", run_id=run_id,
        done=snapshot.done, attempted=snapshot.attempted, target_total=snapshot.target_total,
        combo=snapshot.current_combo.combo if snapshot.current_combo else None,
    )


@subcommand("report", help="重寫 reports/PROGRESS.md 與 dashboard.html（可每 N 秒迴圈）", add_arguments=_add_report_arguments)
def report_command(args: argparse.Namespace) -> int:
    import time

    data_root = load_settings().data_root
    if args.every is None:
        _write_reports_once(data_root, args.run_id)
        return 0
    started = time.monotonic()
    iteration = 0
    try:
        while True:
            try:
                _write_reports_once(data_root, args.run_id)
            except OSError as error:  # 9p／Windows 端暫時鎖檔：記錄後下一輪再試，不讓常駐視窗結束
                print(f"報告寫入失敗，下一輪重試：{error}", flush=True)
                get_logger("report").error(
                    f"報告寫入失敗：{error}", event="report_failed", run_id=args.run_id, errorCode="REPORT_WRITE_FAILED"
                )
            iteration += 1
            if args.count is not None and iteration >= args.count:
                return 0
            time.sleep(max(0.0, started + iteration * args.every - time.monotonic()))
    except KeyboardInterrupt:
        print("report 已停止", flush=True)
        return 0


# ---- aggregate ----
def _add_aggregate_arguments(parser: argparse.ArgumentParser) -> None:
    from pathlib import Path

    parser.add_argument("--run-id", required=True, help="Data Root runs/<run_id>")
    parser.add_argument("--probe", action="store_true", help="只把 probe_results.json 轉成 paper/tables/probe.tex")
    parser.add_argument("--paper-dir", type=Path, default=None, help="輸出 tables/ 與 figures/ 的目錄（預設 Code Root paper/）")


@subcommand("aggregate", help="records → summary.csv、論文 LaTeX 表格與圖（--probe 只產生探針表）", add_arguments=_add_aggregate_arguments)
def aggregate_command(args: argparse.Namespace) -> int:
    from multihop_benchmark.reporting.results_aggregator import AggregateError, aggregate_run

    settings = load_settings()
    run_dir = settings.data_root / "runs" / args.run_id
    paper_dir = args.paper_dir or settings.code_root / "paper"
    logger = get_logger("aggregate")
    try:
        result = aggregate_run(run_dir, paper_dir, probe_only=args.probe)
    except AggregateError as error:
        print(f"aggregate 失敗：{error}")
        logger.error(f"aggregate 失敗：{error}", event="aggregate_failed", run_id=args.run_id, errorCode="AGGREGATE_FAILED")
        return 1
    outputs = [result.summary_path] if result.summary_path else []
    outputs += result.tables + result.figures
    for path in outputs:
        print(f"已寫入 {path}")
    for path in result.removed:
        print(f"已移除 {path}（本 run 沒有 probe_results.json）")
    if result.skipped_lines:
        print(f"警告：略過 {result.skipped_lines} 行無法解析的 records")
    logger.info(
        f"aggregate 完成：{result.record_count} 筆 records，{len(outputs)} 個輸出", event="aggregate_finished",
        run_id=args.run_id, probe_only=args.probe, records=result.record_count, skipped_lines=result.skipped_lines,
        outputs=[str(path) for path in outputs], removed=[str(path) for path in result.removed],
    )
    return 0


# ---- prepare-data ----
def _add_prepare_data_arguments(parser: argparse.ArgumentParser) -> None:
    from multihop_benchmark.datasets.multihop_dataset_loader import DATASET_SPECS

    names = list(DATASET_SPECS)
    parser.add_argument(
        "--datasets", nargs="+", choices=names, default=names, metavar="NAME",
        help=f"要準備的資料集（預設全部：{' '.join(names)}）",
    )
    parser.add_argument("--force", action="store_true", help="輸出已齊全時仍重新產生")


def _jsonl_line_counts(dataset_dir, names) -> dict[str, int]:
    counts = {}
    for name in names:
        with (dataset_dir / f"{name}.jsonl").open(encoding="utf-8") as handle:
            counts[name] = sum(1 for _ in handle)
    return counts


@subcommand(
    "prepare-data", help="下載三集、轉統一 schema、固定種子抽樣並建 corpus（Data Root datasets/）",
    add_arguments=_add_prepare_data_arguments,
)
def prepare_data_command(args: argparse.Namespace) -> int:
    from multihop_benchmark.datasets import multihop_dataset_loader as loader

    datasets_dir = load_settings().data_root / "datasets"
    logger = get_logger("prepare-data")
    logger.info(
        f"prepare-data 開始：{', '.join(args.datasets)}", event="prepare_data_started",
        datasets=args.datasets, force=args.force, seed=loader.SEED,
    )
    loader.write_licenses(datasets_dir)
    failed: list[str] = []
    for dataset in args.datasets:
        dataset_dir = datasets_dir / dataset
        problem = None if args.force else loader.output_problem(dataset_dir)
        if problem is not None and any((dataset_dir / f"{name}.jsonl").exists() for name in loader.OUTPUT_NAMES):
            print(f"{dataset}: 既有輸出不完整（{problem}），重建")
            logger.warning(
                f"{dataset} 既有輸出不完整，重建：{problem}", event="dataset_outputs_invalid",
                dataset=dataset, problem=problem,
            )
        if not args.force and problem is None:
            counts = _jsonl_line_counts(dataset_dir, loader.OUTPUT_NAMES)
            logger.info(
                f"{dataset} 輸出已齊全，略過（--force 可重建）", event="dataset_outputs_exist",
                dataset=dataset, counts=counts,
            )
            note = "（已存在，略過）"
        else:
            try:
                splits = loader.load_raw_dataset(dataset, logger=logger)
            except loader.DatasetDownloadError as error:
                print(f"{dataset}: 失敗：{error}")
                failed.append(dataset)
                continue
            counts = loader.prepare_dataset(dataset, splits, dataset_dir, logger=logger)
            logger.info(
                f"{dataset} 輸出完成", event="dataset_prepared", dataset=dataset, seed=loader.SEED,
                counts=counts, output_dir=str(dataset_dir),
            )
            note = ""
        print(f"{dataset}: " + " ".join(f"{name}={count}" for name, count in counts.items()) + note)

    logger.log(
        logging.ERROR if failed else logging.INFO,
        f"prepare-data 結束，失敗：{failed or '無'}", event="prepare_data_finished",
        datasets=args.datasets, failed=failed, **({"errorCode": "DATASET_DOWNLOAD_FAILED"} if failed else {}),
    )
    return 1 if failed else 0


# ---- build-index ----
def _add_build_index_arguments(parser: argparse.ArgumentParser) -> None:
    from multihop_benchmark.datasets.multihop_dataset_loader import DATASET_SPECS

    names = list(DATASET_SPECS)
    parser.add_argument(
        "--datasets", nargs="+", choices=names, default=names, metavar="NAME",
        help=f"要建索引的資料集（預設全部：{' '.join(names)}）",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda", help="BGE-M3 與 reranker 的裝置（預設 cuda）"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="另以 sample_200 隨機 20 題的支持段標題查詢，寫 indexes/index_report.json（dense top-20／重排 top-3 命中率）",
    )


def _read_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@subcommand(
    "build-index", help="BGE-M3 嵌入三集 corpus、建 FlatIP 索引（Data Root indexes/<dataset>/），可附檢索報告",
    add_arguments=_add_build_index_arguments,
)
def build_index_command(args: argparse.Namespace) -> int:
    import time
    from datetime import datetime, timezone

    from multihop_benchmark import retrieval
    from multihop_benchmark.retrieval import dense_index, passage_reranker

    data_root = load_settings().data_root
    indexes_dir = data_root / "indexes"
    logger = get_logger("build-index")
    logger.info(
        f"build-index 開始：{', '.join(args.datasets)}（{args.device}）", event="build_index_started",
        datasets=args.datasets, device=args.device, report=args.report,
    )
    failed: list[str] = []
    report_error: str | None = None
    entries: dict[str, dict] = {}
    try:  # 任何結果（含模型載入失敗）都以 build_index_finished 收尾
        try:
            embedder = dense_index.load_embedder(args.device)
            reranker = (
                passage_reranker.PassageReranker(model=passage_reranker.load_cross_encoder(args.device))
                if args.report else None
            )
        except Exception as error:  # 模型檔缺漏、CUDA 初始化失敗等：無法處理任何一集
            print(f"模型載入失敗：{error}")
            logger.error(
                f"模型載入失敗：{error}", event="model_load_failed", device=args.device,
                errorCode="MODEL_LOAD_FAILED", exc_info=True,
            )
            failed.extend(args.datasets)
            return 1

        for dataset in args.datasets:  # 每集建索引＋報告是同一個失敗邊界：一集失敗不影響其他集
            dataset_dir = data_root / "datasets" / dataset
            started = time.monotonic()
            stage = "build"
            try:
                corpus = _read_jsonl(dataset_dir / "corpus.jsonl")
                dense_index.build_index(dataset, corpus, args.device, indexes_dir=indexes_dir, embedder=embedder)
                index = dense_index.DenseIndex.load(dataset, indexes_dir=indexes_dir, embedder=embedder)
                if len(index) != len(corpus):
                    raise ValueError(f"向量數 {len(index)} ≠ corpus 行數 {len(corpus)}")
                elapsed = round(time.monotonic() - started, 1)
                print(f"{dataset}: vectors={len(index)} corpus_rows={len(corpus)}（{elapsed}s）")
                logger.info(
                    f"{dataset} 索引完成：{len(index)} 向量", event="index_built", dataset=dataset,
                    vectors=len(index), corpus_rows=len(corpus), device=args.device, elapsed_s=elapsed,
                    output_dir=str(indexes_dir / dataset),
                )
                if reranker is not None:
                    stage = "report"
                    report = retrieval.supporting_title_report(
                        _read_jsonl(dataset_dir / "sample_200.jsonl"), index, reranker
                    )
                    entries[dataset] = {"vectors": len(index), "device": args.device, **report}
                    print(
                        f"{dataset}: 支持段標題查詢 n={report['n']} dense top-{report['dense_k']}"
                        f" 命中率={report['dense_hit_rate']:.3f}"
                        f" 重排 top-{report['top_n']} 命中率={report['rerank_top_n_rate']:.3f}"
                    )
            except Exception as error:  # 讀檔、GPU／推論、報告輸入錯誤皆記錄後續跑下一集
                print(f"{dataset}: 失敗（{stage}）：{error}")
                logger.error(
                    f"{dataset} 建索引失敗（{stage}）：{error}", event="index_build_failed", dataset=dataset,
                    stage=stage, errorCode="INDEX_BUILD_FAILED", exc_info=True,
                )
                failed.append(dataset)

        if entries:
            report_path = indexes_dir / "index_report.json"
            previous = {}
            if report_path.is_file():
                try:
                    previous = json.loads(report_path.read_text("utf-8")).get("datasets", {})
                except (ValueError, AttributeError):
                    previous = {}
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "embedding_model": dense_index.EMBEDDING_MODEL,
                "reranker_model": passage_reranker.RERANKER_MODEL,
                "datasets": {**previous, **entries},
            }
            try:
                report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError as error:
                report_error = str(error)
                print(f"檢索報告寫入失敗：{error}")
                logger.error(
                    f"檢索報告寫入失敗：{error}", event="index_report_failed", path=str(report_path),
                    errorCode="INDEX_REPORT_FAILED", exc_info=True,
                )
            else:
                print(f"已寫入 {report_path}")
                logger.info(
                    f"檢索報告已寫入：{', '.join(entries)}", event="index_report_written", path=str(report_path),
                    rates={name: [entry["dense_hit_rate"], entry["rerank_top_n_rate"]] for name, entry in entries.items()},
                )
    finally:
        ok = not failed and report_error is None
        logger.log(
            logging.INFO if ok else logging.ERROR,
            f"build-index 結束，失敗：{failed or '無'}", event="build_index_finished",
            datasets=args.datasets, failed=failed, report_error=report_error,
            **({} if ok else {"errorCode": "INDEX_REPORT_FAILED" if not failed else "INDEX_BUILD_FAILED"}),
        )
    return 0 if not failed and report_error is None else 1


# ---- make-laya-data ----
def _add_make_laya_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="狀態模擬的固定種子（預設 42）")
    parser.add_argument("--audit-size", type=_non_negative_int, default=100, help="每個切分抽驗筆數（預設 100）")


@subcommand(
    "make-laya-data", help="狀態模擬弱標籤 → Data Root laya_decisions/{train,calibration,test}.jsonl",
    add_arguments=_add_make_laya_data_arguments,
)
def make_laya_data_command(args: argparse.Namespace) -> int:
    from multihop_benchmark.laya_decision import decision_state_builder
    from multihop_benchmark.laya_decision import weak_label_generator as wlg

    data_root = load_settings().data_root
    logger = get_logger("make-laya-data")
    logger.info(f"make-laya-data 開始：seed={args.seed}", event="make_laya_data_started", seed=args.seed)
    try:
        summary = wlg.make_laya_data(data_root, seed=args.seed, tokenizer=decision_state_builder.load_laya_tokenizer())
    except (FileNotFoundError, ValueError) as error:
        print(f"make-laya-data 失敗：{error}")
        logger.error(f"make-laya-data 失敗：{error}", event="make_laya_data_finished", errorCode="LAYA_DATA_FAILED")
        return 1

    types = ("noul", "choice", "score")
    audit_failed = []
    for split, info in summary["splits"].items():
        print(f"[{split}] questions={info['questions']} skipped={info['skipped_questions']} records={info['records']} "
              f"max_state_tokens={info['max_state_tokens']} (limit {decision_state_builder.MAX_STATE_TOKENS}) "
              f"fallback_records={info['fallback_records']} (supporting_only 因 supporting_sentences=null 退回整段) "
              f"sha256={info['sha256']}")
        print(f"  {'state':<16} {'writing':<16} " + " ".join(f"{t:>7}" for t in types))
        for kind in wlg.STATE_KINDS:
            for writing in (wlg.NO_WRITING, *wlg.WRITINGS):
                row = [info["counts"].get((kind, t, writing), 0) for t in types]
                if any(row):
                    print(f"  {kind:<16} {writing:<16} " + " ".join(f"{n:>7}" for n in row))
        print("  labels: " + " | ".join(
            f"{t} " + " ".join(f"{label}={n}" for (typ, label), n in sorted(info["label_counts"].items()) if typ == t)
            for t in types))
        records = [json.loads(line) for line in
                   (data_root / wlg.OUTPUT_DIR_NAME / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        audit = wlg.audit_labels(records, wlg.load_split_questions(data_root, split),
                                 sample_size=args.audit_size, seed=args.seed)
        print(f"  audit: checked={audit['checked']} by_type={audit['by_type']} mismatches={len(audit['mismatches'])}")
        for mismatch in audit["mismatches"][:10]:
            print(f"    {mismatch}")
        if audit["mismatches"]:
            audit_failed.append(split)

    logger.log(
        logging.ERROR if audit_failed else logging.INFO,
        f"make-laya-data 結束，抽驗不符：{audit_failed or '無'}", event="make_laya_data_finished", seed=args.seed,
        records={split: info["records"] for split, info in summary["splits"].items()},
        sha256={split: info["sha256"] for split, info in summary["splits"].items()},
        **({"errorCode": "LAYA_DATA_AUDIT_FAILED"} if audit_failed else {}),
    )
    return 1 if audit_failed else 0


# ---- train-laya ----
def _train_laya_positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"必須 ≥ 1：{value}")
    return number


def _add_train_laya_arguments(parser: argparse.ArgumentParser) -> None:
    from multihop_benchmark.laya_decision.laya_trainer import TrainConfig

    defaults = TrainConfig()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-train", action="store_true", help="模型目錄已有模型時不重新訓練，直接評估；沒有時照常訓練")
    mode.add_argument("--eval-only", action="store_true", help="只評估既有模型；沒有模型時失敗")
    parser.add_argument("--limit", type=_train_laya_positive_int, metavar="N",
                        help="冒煙：train／calibration／test 各只讀前 N 列（報告標示非正式）")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="訓練與評估裝置（預設 cuda）")
    for flag, key, kind in (
        ("--epochs", "epochs", _train_laya_positive_int), ("--micro-batch", "micro_batch", _train_laya_positive_int),
        ("--grad-accum", "grad_accum", _train_laya_positive_int), ("--group-size", "group_size", int),
        ("--lr-encoder", "lr_encoder", float), ("--lr-head", "lr_head", float),
        ("--sigma-start", "sigma_start", float), ("--sigma-end", "sigma_end", float),
        ("--ce-weight", "ce_weight", float), ("--seed", "seed", int),
    ):
        parser.add_argument(flag, dest=key, type=kind, default=getattr(defaults, key),
                            help=f"預設 {getattr(defaults, key)}")


@subcommand(
    "train-laya", help="RLCD 微調 Laya（Runtime Root models/laya_multihop/）並在 held-out test 評估（runs/laya_eval/）",
    add_arguments=_add_train_laya_arguments,
)
def train_laya_command(args: argparse.Namespace) -> int:
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # 必須在 torch 第一次匯入前
    from dataclasses import replace

    from multihop_benchmark.laya_decision import laya_evaluator, laya_trainer
    from multihop_benchmark.laya_decision.laya_decision_client import model_hash

    settings = load_settings()
    decisions = settings.data_root / "laya_decisions"
    paths = {split: decisions / f"{split}.jsonl" for split in ("train", "calibration", "test")}
    model_dir = settings.runtime_root / "models" / "laya_multihop"
    report_dir = settings.data_root / "runs" / "laya_eval"
    summary_path = report_dir / "train_summary.json"
    logger = get_logger("train-laya")
    has_model = (model_dir / "model.safetensors").is_file() and (model_dir / "rl_agent_config.json").is_file()
    will_train = not args.eval_only and not (args.skip_train and has_model)
    logger.info(
        f"train-laya 開始：{'訓練＋評估' if will_train else '只評估'}", event="train_laya_started",
        will_train=will_train, limit=args.limit, device=args.device, model_dir=str(model_dir),
    )

    missing = [str(path) for split, path in paths.items() if (will_train or split == "test") and not path.is_file()]
    if missing:
        print(f"train-laya 失敗：缺少 {', '.join(missing)}；請先執行 make-laya-data")
        logger.error("缺少 Laya 決策資料", event="train_laya_finished", missing=missing, errorCode="LAYA_DATA_MISSING")
        return 1
    if args.eval_only and not has_model:
        print(f"train-laya 失敗：找不到模型 {model_dir}（--eval-only 不會訓練）")
        logger.error("找不到模型", event="train_laya_finished", model_dir=str(model_dir), errorCode="LAYA_MODEL_MISSING")
        return 1

    if will_train:
        config = replace(
            laya_trainer.TrainConfig(), limit=args.limit,
            **{key: getattr(args, key) for key in ("epochs", "micro_batch", "grad_accum", "group_size", "lr_encoder",
                                                   "lr_head", "sigma_start", "sigma_end", "ce_weight", "seed")},
        )
        try:
            train_summary = laya_trainer.train(paths["train"], paths["calibration"], model_dir, config,
                                               device=args.device, logger=logger)
        except (RuntimeError, ValueError, OSError) as error:  # 含 CUDA OOM（torch.OutOfMemoryError 為 RuntimeError）
            print(f"train-laya 訓練失敗：{error}")
            logger.exception(f"訓練失敗：{error}", event="train_laya_finished", errorCode="LAYA_TRAIN_FAILED")
            return 1
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(train_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"訓練摘要已寫入 {summary_path}", event="train_summary_written", path=str(summary_path),
                    elapsed_s=train_summary.get("elapsed_s"))
    else:
        train_summary = json.loads(summary_path.read_text("utf-8")) if summary_path.is_file() else None
        print(f"略過訓練，使用既有模型 {model_dir}")

    finetuned = laya_evaluator.evaluate(model_dir, paths["test"], device=args.device, limit=args.limit)
    zero_shot = laya_evaluator.evaluate(laya_trainer.BASE_MODEL, paths["test"], device=args.device, limit=args.limit)
    verdict = laya_evaluator.judge(finetuned, zero_shot)
    digest = model_hash(model_dir)
    md_path, json_path = laya_evaluator.write_report(
        report_dir, finetuned=finetuned, zero_shot=zero_shot, verdict=verdict, train_summary=train_summary,
        model_dir=model_dir, model_hash=digest, test_path=paths["test"], limit=args.limit,
    )
    status = "PASS" if verdict["passed"] else "FAIL"
    for check in verdict["checks"]:
        value = "—" if check["value"] is None else f"{check['value']:.4f}"
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['name']}: {value}（門檻 {check['threshold']}）")
    for name, metrics in finetuned["by_question"].items():
        base = zero_shot["by_question"].get(name, {})
        print(f"{name}: " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float))
              + f" | zero-shot accuracy={base.get('accuracy', float('nan')):.4f}")
    print(f"p50 latency: 微調 {finetuned['p50_latency_ms']:.1f} ms／zero-shot {zero_shot['p50_latency_ms']:.1f} ms")
    print(f"判定：{status}；報告：{md_path}、{json_path}；model_hash={digest}")
    logger.log(
        logging.INFO if verdict["passed"] else logging.ERROR,
        f"train-laya 評估 {status}", event="eval_finished", status=status, model_hash=digest,
        checks={c["name"]: c["value"] for c in verdict["checks"]}, report=str(md_path), limit=args.limit,
        **({} if verdict["passed"] else {"errorCode": "LAYA_EVAL_THRESHOLD_FAILED"}),
    )
    return 0 if verdict["passed"] else 1


# ---- probe-models ----
def _run_id_argument(value: str) -> str:
    """probe-models／run 共用：run_id 必須是單一路徑片段，在任何寫入前由 argparse 拒絕。"""
    from multihop_benchmark.runs.benchmark_runner import validate_run_id

    try:
        return validate_run_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_probe_models_arguments(parser: argparse.ArgumentParser) -> None:
    from multihop_benchmark.runs.benchmark_runner import MODELS

    parser.add_argument("--run-id", required=True, type=_run_id_argument,
                        help="寫入 Data Root runs/<run_id>/probe_results.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS), metavar="TAG",
                        help=f"Ollama tag（預設五個：{' '.join(MODELS)}）")


@subcommand(
    "probe-models", help="每個模型跑固定 20 題工具呼叫探針 → runs/<run_id>/probe_results.json",
    add_arguments=_add_probe_models_arguments,
)
def probe_models_command(args: argparse.Namespace) -> int:
    from multihop_benchmark.agent.protocols import LLMConnectionError
    from multihop_benchmark.runs import benchmark_runner, probe

    try:
        benchmark_runner.require_unique("models", args.models)
    except ValueError as error:
        print(f"probe-models 參數錯誤：{error}")
        return 2
    settings = load_settings()
    logger = get_logger("probe-models")
    backends = benchmark_runner.production_backends(settings)
    try:
        path = probe.run_probe(args.run_id, args.models, data_root=settings.data_root,
                               llm_factory=backends.llm_factory, ollama=backends.ollama, logger=logger)
    except (probe.ProbeError, LLMConnectionError) as error:
        print(f"probe-models 失敗：{error}")
        if isinstance(error, LLMConnectionError):
            logger.error(f"Ollama 連線失敗：{error}", event="probe_failed", run_id=args.run_id,
                         errorCode="OLLAMA_CONNECTION_FAILED")
        return 1
    results = json.loads(path.read_text(encoding="utf-8"))["models"]
    for model in args.models:
        entry = results[model]
        print(f"{model}: valid_tool_call_rate={entry['valid_tool_call_rate']:.2f}（{entry['valid']}/{entry['n']}）"
              f" mean_latency_ms={entry['mean_latency_ms']:.1f}")
    print(f"已寫入 {path}")
    return 0


# ---- run ----
def _run_positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必須是正整數")
    return number


def _run_positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("必須 > 0")
    return number


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    from multihop_benchmark.runs.benchmark_runner import CONDITIONS, DATASETS, MODELS, TIMEOUT_S

    parser.add_argument("--run-id", required=True, type=_run_id_argument,
                        help="Data Root runs/<run_id>/（同 run_id 重啟即續跑）")
    parser.add_argument("--limit", type=_run_positive_int, metavar="N", help="每組只跑 sample_200 前 N 題（預設全部）")
    parser.add_argument("--models", nargs="+", default=list(MODELS), metavar="TAG",
                        help=f"依此順序分組執行（預設：{' '.join(MODELS)}）")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS), metavar="NAME",
                        help=f"預設全部：{' '.join(DATASETS)}")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS), metavar="NAME",
                        help=f"預設全部：{' '.join(CONDITIONS)}")
    parser.add_argument("--timeout-s", type=_run_positive_float, default=TIMEOUT_S, metavar="SECONDS",
                        help=f"每題逾時秒數（預設 {TIMEOUT_S:g}）")


@subcommand(
    "run", help="依模型分組執行 模型×資料集×條件 組合，每題寫 records（可續跑）", add_arguments=_add_run_arguments,
)
def run_command(args: argparse.Namespace) -> int:
    from multihop_benchmark.agent.protocols import LLMConnectionError
    from multihop_benchmark.runs import benchmark_runner

    try:
        benchmark_runner.validate_selection(args.datasets, args.models, args.conditions)
    except ValueError as error:
        print(f"run 參數錯誤：{error}", flush=True)
        return 2
    settings = load_settings()
    logger = get_logger("run")
    try:
        summary = benchmark_runner.run(
            args.run_id, args.datasets, args.models, args.conditions, args.limit, args.timeout_s,
            data_root=settings.data_root, backends=benchmark_runner.production_backends(settings), logger=logger,
        )
    except (benchmark_runner.RunError, LLMConnectionError) as error:  # 兩者皆已由 runner 記 ERROR
        print(f"run 中止：{error}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("run 已中斷（同 --run-id 重啟即續跑）", flush=True)
        logger.warning("run 被中斷", event="run_interrupted", run_id=args.run_id)
        return 130
    except Exception as error:
        print(f"run 失敗：{type(error).__name__}: {error}", flush=True)
        logger.exception(f"run 失敗：{error}", event="run_failed", run_id=args.run_id, errorCode="RUN_FAILED")
        return 1
    print(f"run 完成：本次執行 {summary['questions_run']} 題 {summary['statuses']}；manifest {summary['manifest']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
