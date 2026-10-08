"""共用正式 Logger：各模組只透過 get_logger(source) 寫 Log。

輸出 JSONL 至 Data Root `logs/`，每行至少含 timestamp / level / event / source /
message，可附 run_id / combo / question_id / errorCode 等欄位（與保留欄位撞名的
關鍵字參數不會覆寫核心欄位，改放在巢狀 `context` 欄位）：

    logger = get_logger("run")
    logger.info("開始組合", event="combo_started", run_id=run_id, combo=combo)

每支獨立程式一個 Log 檔（benchmark / train_laya / prepare_data），由 source 經
SOURCE_LOG_NAMES 對應，未列出的 source 寫入 benchmark；也可用 log_name 指定。
每日午夜輪替，舊檔改名為 `<name>-YYYY-MM-DD.jsonl`，不自動刪除（見 clean-logs）。
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from datetime import date, datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from multihop_benchmark.settings import load_settings

DEFAULT_LOG_NAME = "benchmark"
SOURCE_LOG_NAMES = {
    "run": "benchmark",
    "report": "benchmark",
    "train-laya": "train_laya",
    "prepare-data": "prepare_data",
    "build-index": "prepare_data",
    "make-laya-data": "prepare_data",
}
ROTATED_LOG_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_]+)-(?P<date>\d{4}-\d{2}-\d{2})\.jsonl$")

RESERVED_FIELDS = frozenset({"timestamp", "level", "event", "source", "message", "exception", "context"})
_STANDARD_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}
_loggers: dict[Path, logging.Logger] = {}


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": getattr(record, "event", None) or record.levelname.lower(),
            "source": getattr(record, "source", record.name),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", {})
        line.update({key: value for key, value in fields.items() if key not in RESERVED_FIELDS})
        collisions = {key: value for key, value in fields.items() if key in RESERVED_FIELDS}
        if collisions:
            line["context"] = collisions
        if record.exc_info:
            line["exception"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return json.dumps(line, ensure_ascii=False, default=str)


class BenchmarkLogger(logging.LoggerAdapter):
    """把 event 與其他關鍵字參數轉成 JSONL 欄位，固定帶上 source。"""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        fields = {key: kwargs.pop(key) for key in list(kwargs) if key not in _STANDARD_KWARGS}
        event = fields.pop("event", None)
        kwargs["extra"] = {**kwargs.get("extra", {}), "source": self.extra["source"], "event": event, "fields": fields}
        return msg, kwargs


class _DailyJsonlHandler(TimedRotatingFileHandler):
    def doRollover(self) -> None:
        super().doRollover()
        # 另一程序（run 與 report 共用 benchmark.jsonl）已先輪替時，stdlib 直接 return，
        # 串流仍指向已改名的舊檔（delay=True 時正常輪替後串流必為 None）；
        # 改為關閉串流讓下次寫入重新開啟目前檔，並推進下次輪替時間。
        if self.stream:
            self.stream.close()
            self.stream = None
        self.rolloverAt = self.computeRollover(int(time.time()))


def rotated_log_name(default_name: str) -> str:
    """TimedRotatingFileHandler 的 namer：`benchmark.jsonl.2026-09-29` → `benchmark-2026-09-29.jsonl`。"""
    path = Path(default_name)
    stem, _, day = path.name.rpartition(".")
    return str(path.with_name(f"{Path(stem).stem}-{day}.jsonl"))


def _file_logger(path: Path) -> logging.Logger:
    if path not in _loggers:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = _DailyJsonlHandler(path, when="midnight", backupCount=0, encoding="utf-8", delay=True)
        handler.namer = rotated_log_name
        handler.setFormatter(JsonLineFormatter())
        logger = logging.Logger(f"multihop_benchmark.{path.stem}", logging.INFO)
        logger.addHandler(handler)
        _loggers[path] = logger
    return _loggers[path]


def get_logger(source: str, *, log_name: str | None = None, log_dir: Path | None = None) -> BenchmarkLogger:
    name = log_name or SOURCE_LOG_NAMES.get(source, DEFAULT_LOG_NAME)
    directory = Path(log_dir) if log_dir is not None else load_settings().logs_dir
    return BenchmarkLogger(_file_logger(directory.resolve() / f"{name}.jsonl"), {"source": source})


def delete_rotated_logs(log_dir: Path, older_than_days: int, today: date | None = None) -> list[Path]:
    """刪除檔名日期早於 today - older_than_days 的輪替檔；目前寫入中的檔案不受影響。"""
    cutoff = (today or date.today()) - timedelta(days=older_than_days)
    deleted = []
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        match = ROTATED_LOG_PATTERN.match(path.name)
        if match and date.fromisoformat(match["date"]) < cutoff:
            path.unlink()
            deleted.append(path)
    return deleted
