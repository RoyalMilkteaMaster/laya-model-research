"""60 組實驗排程：依模型分組、每題 `run_question` → 補 `vram_peak_mb`／`finished_at` → `record_store.append`。

排程順序 models → datasets → conditions：同一模型的 12 組連續執行。模型開始前卸載其他常駐模型、以正式
`num_ctx` 預載一次（Log `model_loaded`），組內請求帶 `keep_alive=KEEP_ALIVE` 保持常駐，整組結束以
`keep_alive=0` 卸載。某模型的所有組別都已完成時不載入。

續跑：每組以 `record_store.completed_ids` 取 `status=done` 的題目跳過（Log `resumed skipped=N`）；非 done
（invalid_tool／timeout／oom／error）的題目重跑並 append 新的一行，讀取端以最後一行為準。

錯誤：`status=oom` 以 `num_ctx=OOM_RETRY_NUM_CTX` 的新 client 重跑一次，只寫最後結果；其他非 done 狀態照寫並記
`question_failed`。`LLMConnectionError`（client 已重試 3 次）記 ERROR `run_aborted` 後往外拋，中止整個 run。

`manifest.json` 於啟動時寫入；同一 run_id 重啟時與既有 manifest 合併（見 `_write_manifest`）：網格欄位取保序聯集，
其他設定以第一次為準，每次啟動的實際參數追加在 `launches`。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihop_benchmark.agent import multihop_agent_loop
from multihop_benchmark.agent.protocols import DecisionModel, LLMClient, LLMConnectionError, Retriever
from multihop_benchmark.benchmark_logger import BenchmarkLogger, get_logger
from multihop_benchmark.runs import record_store

DATASETS = ("hotpotqa", "2wiki", "musique")
MODELS = ("qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b")
CONDITIONS = ("recall_llm", "rag_llm", "recall_laya", "rag_laya")
QUESTIONS_FILE = "sample_200.jsonl"
SAMPLE_SIZE = 200

NUM_CTX = 8192
OOM_RETRY_NUM_CTX = 4096
SEED = 42
MAX_HOPS = 4
TIMEOUT_S = 300.0
KEEP_ALIVE = "30m"  # 組內題目間隔遠小於此；換模型時明確卸載
DEVICE = "cuda"  # Spike 結論：BGE-M3、reranker、Laya 皆放 GPU


class RunError(RuntimeError):
    """run 無法開始（模型未安裝、題目檔缺漏等）。"""


def validate_run_id(run_id: Any) -> str:
    """run_id 必須是 Data Root `runs/` 下的單一路徑片段：非空字串、不是 `.`／`..`、不含 `/`、`\\`、NUL。

    run／probe 在任何路徑建構或寫入前呼叫，確保產物只落在 `runs/<run_id>/`（與 record_store 的檔名規則一致）。
    """
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or any(ch in run_id for ch in "/\\\0"):
        raise ValueError(f"run_id 必須是單一路徑片段（非空、不可為 . 或 ..、不可含 / \\ 或 NUL）：{run_id!r}")
    return run_id


def require_unique(name: str, values: Sequence[str]) -> None:
    """同一次 launch 的選擇不可重複，否則同一組會排程兩次、寫出重複 question_id。"""
    duplicated = sorted({value for value in values if list(values).count(value) > 1})
    if duplicated:
        raise ValueError(f"{name} 重複：{', '.join(duplicated)}")


def validate_selection(datasets: Sequence[str], models: Sequence[str], conditions: Sequence[str]) -> None:
    """datasets／conditions 必須在允許清單內；三者皆不可重複。"""
    for name, values, allowed in (("dataset", datasets, DATASETS), ("condition", conditions, CONDITIONS)):
        unknown = [value for value in values if value not in allowed]
        if unknown:
            raise ValueError(f"未知 {name}：{', '.join(unknown)}（允許 {', '.join(allowed)}）")
    for name, values in (("datasets", datasets), ("models", models), ("conditions", conditions)):
        require_unique(name, values)


def combo_name(dataset: str, model: str, condition: str) -> str:
    return f"{dataset}__{model.replace(':', '-')}__{condition}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---- 外部邊界 ----------------------------------------------------------------


class OllamaControl:
    """Ollama 模型管理（清單與 digest、預載、卸載、常駐清單）；連線失敗重試後拋 `LLMConnectionError`。"""

    def __init__(self, host: str, *, keep_alive: str | float = KEEP_ALIVE,
                 retry_delays_s: Sequence[float] = (1.0, 2.0, 4.0)) -> None:
        import ollama

        self.host = host
        self.keep_alive = keep_alive
        self.retry_delays_s = tuple(retry_delays_s)
        self._client = ollama.Client(host=host)

    def _call(self, what: str, fn: Callable[[], Any]) -> Any:
        import httpx

        for attempt in range(len(self.retry_delays_s) + 1):
            try:
                return fn()
            except (ConnectionError, httpx.TransportError) as exc:
                if attempt == len(self.retry_delays_s):
                    raise LLMConnectionError(f"Ollama {self.host} {what} 連線失敗，已重試 {attempt} 次：{exc}") from exc
                time.sleep(self.retry_delays_s[attempt])
        raise AssertionError("unreachable")

    def digests(self) -> dict[str, str]:
        return {m.model: m.digest for m in self._call("list", self._client.list).models}

    def version(self) -> str | None:
        import httpx

        try:
            return self._call("version", lambda: httpx.get(f"{self.host}/api/version", timeout=10).json()["version"])
        except (LLMConnectionError, ValueError, KeyError):
            return None

    def running(self) -> list[str]:
        return [m.model for m in self._call("ps", self._client.ps).models]

    def load(self, model: str, num_ctx: int) -> None:
        options = {"num_ctx": num_ctx, "temperature": 0, "seed": SEED}  # 與正式請求一致，避免第一題重新載入
        self._call("load", lambda: self._client.generate(model=model, prompt="", keep_alive=self.keep_alive,
                                                         options=options))

    def unload(self, model: str) -> None:
        self._call("unload", lambda: self._client.generate(model=model, prompt="", keep_alive=0))


def git_commit(code_root: Path) -> str | None:
    """HEAD commit；工作樹有未 commit 變更時加上 `-dirty`。取不到時回 None。"""
    try:
        head = subprocess.run(["git", "-C", str(code_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(code_root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{head}-dirty" if dirty else head


@dataclass
class Backends:
    """執行器的外部依賴；測試注入假物件，正式執行用 `production_backends`。"""

    llm_factory: Callable[[str, int], LLMClient]  # (model tag, num_ctx)
    ollama: Any  # OllamaControl 介面：digests／version／running／load／unload
    decision_factory: Callable[[], DecisionModel]
    retriever_factory: Callable[[str], Retriever]
    laya_model_hash: Callable[[], str | None]
    git_commit: Callable[[], str | None]
    sampler_factory: Callable[[], Any] = field(default=None)  # start()／stop() -> MiB|None／error
    retrieval_device: str = DEVICE
    laya_device: str = DEVICE
    laya_model_dir: str | None = None

    def __post_init__(self) -> None:
        if self.sampler_factory is None:
            from multihop_benchmark.evaluation.gpu_memory_sampler import GpuMemorySampler

            self.sampler_factory = GpuMemorySampler


def production_backends(settings: Any, *, device: str = DEVICE) -> Backends:
    """正式依賴：Ollama client、微調 Laya、共用 BGE-M3／reranker 的三集 Retriever、pynvml 取樣。"""
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # 必須在 torch 第一次匯入前
    from multihop_benchmark.agent.ollama_llm_client import OllamaLLMClient

    laya_dir = settings.runtime_root / "models" / "laya_multihop"
    indexes_dir = settings.data_root / "indexes"
    shared: dict[str, Any] = {}

    def retriever_factory(dataset: str) -> Retriever:
        from multihop_benchmark.retrieval import Retriever as StackRetriever
        from multihop_benchmark.retrieval.dense_index import DenseIndex, load_embedder
        from multihop_benchmark.retrieval.passage_reranker import PassageReranker

        if not shared:  # 三集共用一份 BGE-M3 與 reranker，只佔一次 VRAM
            shared["embedder"] = load_embedder(device)
            shared["reranker"] = PassageReranker(device)
        index = DenseIndex.load(dataset, indexes_dir=indexes_dir, embedder=shared["embedder"])
        return StackRetriever(index, shared["reranker"])

    def decision_factory() -> DecisionModel:
        from multihop_benchmark.laya_decision.laya_decision_client import LayaDecisionClient

        return LayaDecisionClient(laya_dir, device=device)

    def laya_hash() -> str | None:
        from multihop_benchmark.laya_decision.laya_decision_client import model_hash

        return model_hash(laya_dir) if laya_dir.is_dir() else None

    return Backends(
        llm_factory=lambda model, num_ctx: OllamaLLMClient(
            model, settings.ollama_host, num_ctx=num_ctx, seed=SEED, keep_alive=KEEP_ALIVE),
        ollama=OllamaControl(settings.ollama_host),
        decision_factory=decision_factory,
        retriever_factory=retriever_factory,
        laya_model_hash=laya_hash,
        git_commit=lambda: git_commit(settings.code_root),
        retrieval_device=device,
        laya_device=device,
        laya_model_dir=str(laya_dir),
    )


# ---- 執行 --------------------------------------------------------------------


def load_questions(data_root: Path, dataset: str, limit: int | None) -> list[dict[str, Any]]:
    path = Path(data_root) / "datasets" / dataset / QUESTIONS_FILE
    if not path.is_file():
        raise RunError(f"找不到題目檔 {path}；請先執行 prepare-data")
    questions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                questions.append(json.loads(line))
                if limit is not None and len(questions) >= limit:
                    break
    return questions


def _probe_summary(run_dir: Path, models: Sequence[str]) -> dict[str, Any] | None:
    path = run_dir / "probe_results.json"
    try:
        probes = json.loads(path.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError, AttributeError):
        return None
    keys = ("valid_tool_call_rate", "n", "mean_latency_ms")
    return {model: ({key: probes[model].get(key) for key in keys} if isinstance(probes.get(model), dict) else None)
            for model in models}


GRID_KEYS = ("datasets", "models", "conditions")
LAUNCH_KEYS = (*GRID_KEYS, "limit", "questions_per_combo", "model_digests", "loop", "seed", "git_commit",
               "ollama_version", "laya_model_hash")
# None 代表「不適用／取不到」的欄位：既有為 None 時以本次值補上。limit 的 None 是「全部題目」，不在此列。
FILL_IN_KEYS = ("laya_model_hash", "laya_model_dir", "laya_device", "git_commit", "ollama_version")


def _names(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _union(*lists: list[str]) -> list[str]:
    merged: list[str] = []
    for names in lists:
        merged += [name for name in names if name not in merged]
    return merged


def _probe_digests(run_dir: Path) -> dict[str, str]:
    try:
        probes = json.loads((run_dir / "probe_results.json").read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError, AttributeError):
        return {}
    return {model: entry["digest"] for model, entry in probes.items()
            if isinstance(entry, dict) and isinstance(entry.get("digest"), str) and entry["digest"]}


def _write_manifest(run_dir: Path, run_id: str, fields: dict[str, Any], logger: BenchmarkLogger,
                    installed_digests: Mapping[str, str]) -> Path:
    """寫入（或與既有合併）`manifest.json`；一個 run_id 可分多次啟動、每次只跑網格的一部分。

    - `datasets`／`models`／`conditions`：既有頂層 → 既有 `launches` 各次 → 本次，保序聯集（新名稱附加在後）；
      納入 `launches` 可補回舊版覆寫時遺失的名稱。report 以此建網格，因此永遠是整個 run 的完整網格。
    - `model_digests`：既有條目保留，新 tag 附加。合併後仍缺 digest 的 tag（例如只從舊 launches 補回、本次未啟動）
      依序由本 run 的 probe_results.json（探針當時記下的 digest，最接近實際使用的版本）與目前 `ollama list`
      （`installed_digests`）補回；都補不到時保留 null 並記 WARNING `manifest_digest_missing`，不中止啟動。
    - `probe_summary` 依合併後的 models 重新讀 probe_results.json。
    - 其他設定（`questions_per_combo`、`limit`、`loop`、`seed`、`git_commit`、Laya hash…）以第一次為準：
      `FILL_IN_KEYS` 的既有值為 None 時補上本次值（例如第一次沒有 Laya 條件）；其餘與本次不同時保留既有值，
      記 WARNING `manifest_settings_kept`（含 keys）。不拒絕啟動，
      以免補跑被無關差異（例如新 commit）擋下；是否混用設定由 `launches` 追溯。
    - `launches` 追加本次的實際參數（含 digests、loop、questions_per_combo）。
    """
    path = run_dir / "manifest.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
    started_at = utc_now_iso()
    launches = [launch for launch in previous.get("launches", []) if isinstance(launch, dict)]
    launch = {"started_at": started_at, **{key: fields.get(key) for key in LAUNCH_KEYS}}

    merged = dict(fields)
    kept: list[str] = []
    if previous:
        for key in GRID_KEYS:
            merged[key] = _union(_names(previous.get(key)), *(_names(item.get(key)) for item in launches), fields[key])
        digests = dict(previous.get("model_digests") or {})
        for model, digest in fields["model_digests"].items():
            if digests.get(model) not in (None, digest):
                kept.append(f"model_digests.{model}")
            digests.setdefault(model, digest)
        fallbacks = (_probe_digests(run_dir), installed_digests)
        for model in merged["models"]:
            if not digests.get(model):
                digests[model] = next((source[model] for source in fallbacks if source.get(model)), None)
        merged["model_digests"] = {model: digests[model] for model in merged["models"]}
        missing_digests = [model for model in merged["models"] if not digests[model]]
        if missing_digests:
            logger.warning(f"manifest 無法補回模型 digest：{', '.join(missing_digests)}（保留 null）",
                           event="manifest_digest_missing", run_id=run_id, models=missing_digests,
                           errorCode="MANIFEST_DIGEST_MISSING")
        for key, value in fields.items():
            if key in (*GRID_KEYS, "model_digests", "probe_summary") or key not in previous:
                continue
            fill_in = key in FILL_IN_KEYS
            if fill_in and (previous[key] is None or value is None):
                merged[key] = value if previous[key] is None else previous[key]
                continue
            if previous[key] != value:
                kept.append(key)
            merged[key] = previous[key]
        merged["probe_summary"] = _probe_summary(run_dir, merged["models"])
    if kept:
        logger.warning(f"manifest 保留第一次的設定（本次不同：{', '.join(kept)}），本次參數記於 launches",
                       event="manifest_settings_kept", run_id=run_id, keys=kept,
                       errorCode="MANIFEST_SETTINGS_DIFFER")
    manifest = {
        "run_id": run_id,
        "created_at": previous.get("created_at", started_at),
        "started_at": started_at,
        **merged,
        "launches": [*launches, launch],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".manifest.json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def run(
    run_id: str,
    datasets: Sequence[str] = DATASETS,
    models: Sequence[str] = MODELS,
    conditions: Sequence[str] = CONDITIONS,
    limit: int | None = None,
    timeout_s: float = TIMEOUT_S,
    *,
    data_root: Path,
    backends: Backends,
    max_hops: int = MAX_HOPS,
    logger: BenchmarkLogger | None = None,
) -> dict[str, Any]:
    """執行（或續跑）一個 run；回傳 `{questions_run, statuses, manifest}`。行為見模組說明。"""
    validate_run_id(run_id)
    datasets, models, conditions = list(datasets), list(models), list(conditions)
    validate_selection(datasets, models, conditions)
    if limit is not None and limit < 1:
        raise ValueError(f"limit 必須 ≥ 1：{limit}")
    logger = logger or get_logger("run")
    where: dict[str, str] = {}  # 目前的模型／組別／題目，寫入中止 Log
    try:
        return _run(run_id, datasets, models, conditions, limit, timeout_s, Path(data_root), backends, max_hops,
                    logger, where)
    except LLMConnectionError as exc:
        logger.error(f"Ollama 連線失敗，中止整個 run：{exc}", event="run_aborted", run_id=run_id,
                     errorCode="OLLAMA_CONNECTION_FAILED", **where)
        raise


def _run(
    run_id: str, datasets: list[str], models: list[str], conditions: list[str], limit: int | None, timeout_s: float,
    data_root: Path, backends: Backends, max_hops: int, logger: BenchmarkLogger, where: dict[str, str],
) -> dict[str, Any]:
    run_dir = data_root / "runs" / run_id
    try:
        questions = {dataset: load_questions(data_root, dataset, limit) for dataset in datasets}
    except RunError as error:
        logger.error(str(error), event="run_failed", run_id=run_id, errorCode="QUESTIONS_MISSING")
        raise
    digests = backends.ollama.digests()
    missing = [model for model in models if model not in digests]
    if missing:
        message = f"Ollama 未安裝模型：{', '.join(missing)}（請先 ollama pull）"
        logger.error(message, event="run_failed", run_id=run_id, errorCode="MODEL_MISSING", missing=missing)
        raise RunError(message)

    uses_laya = any(c.endswith("_laya") for c in conditions)
    uses_rag = any(c.startswith("rag_") for c in conditions)
    commit = backends.git_commit()
    manifest_fields = {
        "datasets": datasets,
        "models": models,
        "conditions": conditions,
        "questions_per_combo": min(limit or SAMPLE_SIZE, *(len(q) for q in questions.values())),
        "limit": limit,
        "model_digests": {model: digests[model] for model in models},
        "ollama_version": backends.ollama.version(),
        "loop": {
            "max_hops": max_hops, "timeout_s": timeout_s, "num_ctx": NUM_CTX, "oom_retry_num_ctx": OOM_RETRY_NUM_CTX,
            "temperature": 0, "think": False, "keep_alive": KEEP_ALIVE,
            "sufficient_threshold": multihop_agent_loop.SUFFICIENT_THRESHOLD,
            "rag_top_n": multihop_agent_loop.RAG_TOP_N, "max_evidence": multihop_agent_loop.MAX_EVIDENCE,
        },
        "seed": SEED,
        "git_commit": commit,
        "laya_model_hash": backends.laya_model_hash() if uses_laya else None,
        "laya_model_dir": backends.laya_model_dir if uses_laya else None,
        "laya_device": backends.laya_device if uses_laya else None,
        "retrieval_device": backends.retrieval_device,
        "probe_summary": _probe_summary(run_dir, models),
    }
    manifest_path = _write_manifest(run_dir, run_id, manifest_fields, logger, digests)
    logger.info(f"run 開始：{len(models)} 模型 × {len(datasets)} 資料集 × {len(conditions)} 條件",
                event="run_started", run_id=run_id, datasets=datasets, models=models, conditions=conditions,
                limit=limit, timeout_s=timeout_s, manifest=str(manifest_path))

    # 檢索堆疊與 Laya 先上 GPU（Spike 的載入順序），再依序載入 LLM。
    decision = backends.decision_factory() if uses_laya else None
    retrievers = {dataset: backends.retriever_factory(dataset) for dataset in datasets} if uses_rag else {}

    statuses: Counter[str] = Counter()
    sampler_warned = False
    for model in models:
        where.clear()
        where["model"] = model
        plan = []
        for dataset in datasets:
            for condition in conditions:
                combo = combo_name(dataset, model, condition)
                done = record_store.completed_ids(run_id, combo, data_root=data_root)
                todo = [q for q in questions[dataset] if q["id"] not in done]
                skipped = len(questions[dataset]) - len(todo)
                if skipped:
                    logger.info(f"{combo} 續跑：跳過 {skipped} 題已完成", event="resumed", run_id=run_id,
                                combo=combo, skipped=skipped, remaining=len(todo))
                if todo:
                    plan.append((dataset, condition, combo, todo))
        if not plan:
            logger.info(f"{model} 的組別皆已完成，不載入", event="model_skipped", run_id=run_id, model=model)
            continue

        for other in backends.ollama.running():
            if other != model:
                backends.ollama.unload(other)
                logger.info(f"卸載其他常駐模型 {other}", event="model_unloaded", run_id=run_id, model=other)
        started = time.perf_counter()
        backends.ollama.load(model, NUM_CTX)
        load_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(f"模型已載入 {model}（{load_ms} ms）", event="model_loaded", run_id=run_id, model=model,
                    digest=digests[model], num_ctx=NUM_CTX, keep_alive=KEEP_ALIVE, load_ms=load_ms)
        llm = backends.llm_factory(model, NUM_CTX)

        for dataset, condition, combo, todo in plan:
            evidence_source, controller = condition.split("_")
            logger.info(f"{combo} 開始：{len(todo)} 題", event="combo_started", run_id=run_id, combo=combo,
                        model=model, dataset=dataset, condition=condition, todo=len(todo))
            counts: Counter[str] = Counter()
            for question in todo:
                where.update(combo=combo, question_id=question["id"])
                sampler = backends.sampler_factory()
                sampler.start()
                try:
                    record = _run_one(question, evidence_source, controller, llm, decision,
                                      retrievers.get(dataset), max_hops, timeout_s, run_id, dataset, model,
                                      condition, combo, backends, logger)
                finally:
                    peak = sampler.stop()
                if sampler.error and not sampler_warned:
                    sampler_warned = True
                    logger.warning(f"VRAM 取樣不可用：{sampler.error}", event="vram_sampler_unavailable",
                                   run_id=run_id, errorCode="VRAM_SAMPLER_UNAVAILABLE")
                record["vram_peak_mb"] = peak
                record["finished_at"] = utc_now_iso()
                record_store.append(run_id, combo, record, data_root=data_root)
                counts[record["status"]] += 1
                if record["status"] != "done":
                    logger.warning(f"{combo} {question['id']} {record['status']}：{record.get('error', '')}",
                                   event="question_failed", run_id=run_id, combo=combo, question_id=question["id"],
                                   errorCode=f"QUESTION_{record['status'].upper()}", status=record["status"])
            statuses.update(counts)
            logger.info(f"{combo} 完成：{dict(counts)}", event="combo_finished", run_id=run_id, combo=combo,
                        model=model, dataset=dataset, condition=condition, counts=dict(counts))

        where.pop("combo", None)
        where.pop("question_id", None)
        backends.ollama.unload(model)
        logger.info(f"模型已卸載 {model}", event="model_unloaded", run_id=run_id, model=model)

    total = sum(statuses.values())
    logger.info(f"run 完成：本次執行 {total} 題 {dict(statuses)}", event="run_finished", run_id=run_id,
                questions_run=total, statuses=dict(statuses))
    return {"questions_run": total, "statuses": dict(statuses), "manifest": str(manifest_path)}


def _run_one(
    question: Mapping[str, Any], evidence_source: str, controller: str, llm: LLMClient,
    decision: DecisionModel | None, retriever: Retriever | None, max_hops: int, timeout_s: float,
    run_id: str, dataset: str, model: str, condition: str, combo: str, backends: Backends, logger: BenchmarkLogger,
) -> dict[str, Any]:
    def attempt(client: LLMClient) -> dict[str, Any]:
        return multihop_agent_loop.run_question(
            question, evidence_source, controller, client,
            decision if controller == "laya" else None, retriever if evidence_source == "rag" else None,
            max_hops, timeout_s, run_id=run_id, dataset=dataset, model=model, condition=condition,
        )

    record = attempt(llm)
    if record["status"] == "oom":
        logger.warning(f"{combo} {question['id']} OOM，以 num_ctx={OOM_RETRY_NUM_CTX} 重試一次：{record.get('error')}",
                       event="oom_retry", run_id=run_id, combo=combo, question_id=question["id"],
                       errorCode="QUESTION_OOM_RETRY", num_ctx=OOM_RETRY_NUM_CTX)
        record = attempt(backends.llm_factory(model, OOM_RETRY_NUM_CTX))
    return record
