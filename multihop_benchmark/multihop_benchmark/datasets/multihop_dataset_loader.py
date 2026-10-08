"""載入 HotpotQA／2WikiMultihopQA／MuSiQue，轉成統一 schema 並以固定種子抽樣。

統一 schema（每行一題）：

    {id, question, answer, answer_aliases[],
     paragraphs[{pid, title, text, is_supporting, supporting_sentences}]}

- HotpotQA、2Wiki：段落 = context 的一個 title＋其句子串接；`is_supporting` 為該 title
  是否出現在 `supporting_facts.title`；無 aliases，`answer_aliases` 為空陣列。
- MuSiQue：直接使用 `paragraphs[].is_supporting` 與 `answer_aliases`。
- `supporting_sentences`（鍵固定存在）：HotpotQA、2Wiki 的支持段落為 `supporting_facts` 中該 title
  各 `sent_id` 的句子，依原始句序、只去除頭尾空白；`sent_id` 超出句數者略過並記 WARNING
  `supporting_sentence_out_of_range`；非支持段落為 `[]`。MuSiQue 沒有句子層級資訊，一律 `null`。
- 句子串接：前一句結尾或下一句開頭已有空白時直接相接，否則補一個空白；最後去除頭尾空白。
- `pid` = sha256(`title` + "\\n" + `text`) 的前 16 個十六進位字元：同一資料集內同內容必同 pid，
  可跨題去重。同一題內 pid 重複的段落（2Wiki 偶有重複干擾段落）只保留第一次出現，
  `is_supporting` 取聯集。
- 完成 manifest：五檔原子寫入並通過結構檢查後，最後才原子寫入 `prepare_manifest.json`（schema／
  generator 版本、seed、UTC 建立時間、HF 來源、五檔 sha256 與行數）；重建開始時先刪除舊 manifest。
  `output_problem` 先比對 manifest 與實際 sha256，再做結構檢查。
- 抽樣：id 先排序再用 `random.Random(SEED)`，與 HF 快取狀態、列順序無關；validation 抽
  `sample_200`，train 一次抽 1000 題依序切成互斥的 `train_800`／`calibration_100`／`test_100`。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multihop_benchmark.datasets.corpus_builder import build_corpus

DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_BACKOFF_S = 10.0
SEED = 42
TEST_SAMPLE_NAME = "sample_200"
TEST_SAMPLE_SIZE = 200
TRAIN_SPLIT_SIZES = {"train_800": 800, "calibration_100": 100, "test_100": 100}
CORPUS_NAME = "corpus"
OUTPUT_NAMES = (TEST_SAMPLE_NAME, *TRAIN_SPLIT_SIZES, CORPUS_NAME)
QUESTION_KEYS = ["id", "question", "answer", "answer_aliases", "paragraphs"]
PARAGRAPH_KEYS = ["pid", "title", "text", "is_supporting", "supporting_sentences"]
CORPUS_KEYS = ["pid", "title", "text"]
MANIFEST_NAME = "prepare_manifest.json"
SCHEMA_VERSION = "2"  # 2 = 段落含 supporting_sentences
GENERATOR = {"module": __name__, "version": "1"}  # 轉換或抽樣邏輯改變時遞增

Record = dict[str, Any]


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    config: str
    convert: Callable[[str, Mapping[str, Any], Any], Record]


def paragraph_id(title: str, text: str) -> str:
    return hashlib.sha256(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]


def _join_sentences(sentences: list[str]) -> str:
    text = ""
    for sentence in sentences:
        if text and not text[-1].isspace() and sentence and not sentence[0].isspace():
            text += " "
        text += sentence
    return text.strip()


def _paragraph(title: str, text: str, is_supporting: bool, supporting_sentences: list[str] | None) -> Record:
    return {
        "pid": paragraph_id(title, text),
        "title": title,
        "text": text,
        "is_supporting": bool(is_supporting),
        "supporting_sentences": supporting_sentences,
    }


def _record(raw: Mapping[str, Any], answer_aliases: list[str], paragraphs: list[Record]) -> Record:
    unique: dict[str, Record] = {}
    for paragraph in paragraphs:
        if paragraph["pid"] in unique:
            kept = unique[paragraph["pid"]]
            kept["is_supporting"] |= paragraph["is_supporting"]
            kept["supporting_sentences"] = kept["supporting_sentences"] or paragraph["supporting_sentences"]
        else:
            unique[paragraph["pid"]] = paragraph
    return {
        "id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "answer_aliases": list(answer_aliases),
        "paragraphs": list(unique.values()),
    }


def _convert_context_format(dataset: str, raw: Mapping[str, Any], logger: Any) -> Record:
    """HotpotQA（distractor）與 2Wiki 共用的 context + supporting_facts 格式。"""
    sent_ids_by_title: dict[str, set[int]] = {}
    facts = raw["supporting_facts"]
    for title, sent_id in zip(facts["title"], facts["sent_id"], strict=True):
        sent_ids_by_title.setdefault(title, set()).add(sent_id)

    warned: set[tuple[str, int]] = set()
    paragraphs = []
    context = raw["context"]
    for title, sentences in zip(context["title"], context["sentences"], strict=True):
        sent_ids = sent_ids_by_title.get(title)
        supporting_sentences = []
        for sent_id in sorted(sent_ids or ()):
            if 0 <= sent_id < len(sentences):
                supporting_sentences.append(sentences[sent_id].strip())
            elif logger is not None and (title, sent_id) not in warned:
                warned.add((title, sent_id))
                logger.warning(
                    f"{raw['id']} 的 supporting_facts sent_id {sent_id} 超出「{title}」的 {len(sentences)} 句，略過",
                    event="supporting_sentence_out_of_range", dataset=dataset, question_id=raw["id"],
                    title=title, sent_id=sent_id, sentence_count=len(sentences),
                )
        paragraphs.append(_paragraph(title, _join_sentences(sentences), sent_ids is not None, supporting_sentences))
    return _record(raw, [], paragraphs)


def _convert_musique(dataset: str, raw: Mapping[str, Any], logger: Any) -> Record:
    paragraphs = [
        _paragraph(p["title"], p["paragraph_text"].strip(), p["is_supporting"], None) for p in raw["paragraphs"]
    ]
    return _record(raw, raw["answer_aliases"] or [], paragraphs)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec("hotpotqa/hotpot_qa", "distractor", _convert_context_format),
    "2wiki": DatasetSpec("framolfese/2WikiMultihopQA", "default", _convert_context_format),
    "musique": DatasetSpec("dgslibisey/MuSiQue", "default", _convert_musique),
}


def convert_question(dataset: str, raw: Mapping[str, Any], *, logger: Any = None) -> Record:
    """logger 用於記錄資料異常（例如 sent_id 越界）；None 時不記錄。"""
    return DATASET_SPECS[dataset].convert(dataset, raw, logger)


class DatasetDownloadError(RuntimeError):
    """重試用盡仍無法下載或載入資料集。"""


def _default_builder_factory(repo: str, config: str) -> Any:
    from datasets import load_dataset_builder  # 重依賴延後匯入；使用 HF_HOME 快取

    return load_dataset_builder(repo, config)


def load_raw_dataset(
    dataset: str,
    *,
    logger: Any,
    builder_factory: Callable[[str, str], Any] | None = None,
    max_attempts: int = DOWNLOAD_ATTEMPTS,
    backoff_s: float = DOWNLOAD_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """經 HF 快取載入原始資料集的各 split；失敗以指數退避重試。

    已完成的 HF 快取（`<cache_dir>/dataset_info.json` 存在，與 datasets 判定相同）記
    `dataset_cache_hit` 且不重新下載；斷網時 datasets 會退回使用最近一次的快取版本。
    """
    spec = DATASET_SPECS[dataset]
    builder_factory = builder_factory or _default_builder_factory
    fields = {"dataset": dataset, "repo": spec.repo, "config": spec.config}
    for attempt in range(1, max_attempts + 1):
        try:
            builder = builder_factory(spec.repo, spec.config)
            cache_dir = str(builder.cache_dir)
            if (Path(cache_dir) / "dataset_info.json").is_file():
                logger.info(f"{dataset} 使用 HF 快取", event="dataset_cache_hit", cache_dir=cache_dir, **fields)
            else:
                logger.info(f"{dataset} 開始下載", event="dataset_download_started", attempt=attempt, **fields)
            builder.download_and_prepare()
            splits = builder.as_dataset()
        except Exception as error:  # 網路錯誤型別眾多（HTTP、Xet、DNS），一律重試
            if attempt == max_attempts:
                logger.error(
                    f"{dataset} 下載失敗（{max_attempts} 次）：{error}",
                    event="dataset_download_failed", errorCode="DATASET_DOWNLOAD_FAILED", attempt=attempt, **fields,
                )
                raise DatasetDownloadError(f"{spec.repo} 下載失敗（已試 {max_attempts} 次）：{error}") from error
            delay = backoff_s * 2 ** (attempt - 1)
            logger.warning(
                f"{dataset} 第 {attempt} 次下載失敗，{delay:g} 秒後重試：{error}",
                event="dataset_download_retry", attempt=attempt, delay_s=delay, error=str(error), **fields,
            )
            sleep(delay)
            continue
        logger.info(
            f"{dataset} 載入完成", event="dataset_loaded", cache_dir=cache_dir,
            splits={name: len(split) for name, split in splits.items()}, **fields,
        )
        return splits
    raise AssertionError("unreachable")


def sample_ids(ids: Iterable[str], k: int, *, seed: int = SEED) -> list[str]:
    """先依 id 排序再用 random.Random(seed) 抽 k 個，結果與輸入順序、快取狀態無關。"""
    pool = sorted(ids)
    if k > len(pool):
        raise ValueError(f"題數不足：需要 {k} 題，只有 {len(pool)} 題")
    return random.Random(seed).sample(pool, k)


def split_train_ids(
    ids: Iterable[str], sizes: Mapping[str, int] = TRAIN_SPLIT_SIZES, *, seed: int = SEED
) -> dict[str, list[str]]:
    """一次抽出 sum(sizes) 題，依 sizes 順序切成互斥的區段。"""
    sampled = sample_ids(ids, sum(sizes.values()), seed=seed)
    splits: dict[str, list[str]] = {}
    start = 0
    for name, size in sizes.items():
        splits[name] = sampled[start : start + size]
        start += size
    return splits


LICENSES_MARKDOWN = """\
# 資料集授權

由 `multihop-benchmark prepare-data` 產生；授權於 2026-09-30 依下列原始來源查證。

| 資料集 | Hugging Face 來源 | 授權 | 查證來源 |
|---|---|---|---|
| HotpotQA（distractor） | `hotpotqa/hotpot_qa` | CC BY-SA 4.0 | https://hotpotqa.github.io/ ；HF dataset card metadata `license: cc-by-sa-4.0` |
| 2WikiMultihopQA | `framolfese/2WikiMultihopQA` | Apache-2.0 | https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE （GitHub API `spdx_id: Apache-2.0`）；HF card 內文聲明沿用 Apache License 2.0（無 metadata 欄位） |
| MuSiQue（musique-ans） | `dgslibisey/MuSiQue` | CC BY 4.0 | https://github.com/StonyBrookNLP/musique/blob/main/LICENSE （GitHub API `spdx_id: CC-BY-4.0`）；HF 鏡像無 dataset card 與授權標示 |

本目錄的 `sample_200`／`train_800`／`calibration_100`／`test_100`／`corpus` 為上述資料集的抽樣與
格式轉換（衍生物），沿用各自原授權；HotpotQA 衍生物依 CC BY-SA 4.0 以相同授權分享。
"""


def write_licenses(datasets_dir: Path) -> Path:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    path = datasets_dir / "LICENSES.md"
    path.write_text(LICENSES_MARKDOWN, encoding="utf-8", newline="\n")
    return path


def _convert_selected(dataset: str, split: Any, ids: list[str], logger: Any) -> list[Record]:
    """只轉換抽中的題目；split 支援 `split["id"]` 欄與 `split[i]` 列（HF Dataset）。"""
    index_of = {question_id: index for index, question_id in enumerate(split["id"])}
    return [convert_question(dataset, split[index_of[question_id]], logger=logger) for question_id in ids]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """UTF-8、LF、ensure_ascii=False，先寫暫存檔再改名，避免中斷留下半截檔。"""
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _read_jsonl_rows(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} 第 {number} 行無法解析：{error}") from error
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_problem(output_dir: Path, seed: int, names: tuple[str, ...]) -> str | None:
    path = output_dir / MANIFEST_NAME
    if not path.is_file():
        return f"缺少 {MANIFEST_NAME}"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return f"{MANIFEST_NAME} 無法解析：{error}"
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return f"{MANIFEST_NAME} 格式不符"
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return f"{MANIFEST_NAME} 的 schema_version {manifest.get('schema_version')!r} 不是 {SCHEMA_VERSION!r}"
    if manifest.get("generator") != GENERATOR or manifest.get("seed") != seed:
        return f"{MANIFEST_NAME} 的 generator 或 seed 與目前程式不同"
    files = manifest["files"]
    if set(files) != {f"{name}.jsonl" for name in names}:
        return f"{MANIFEST_NAME} 列出的檔案與輸出不符"
    for name in names:
        file_path = output_dir / f"{name}.jsonl"
        if not file_path.is_file():
            return f"缺少 {file_path.name}"
        entry = files[file_path.name]
        if not isinstance(entry, dict) or entry.get("sha256") != _sha256(file_path):
            return f"{file_path.name} 的 sha256 與 {MANIFEST_NAME} 不符"
    return None


def output_problem(
    output_dir: Path,
    *,
    seed: int = SEED,
    sample_size: int = TEST_SAMPLE_SIZE,
    train_sizes: Mapping[str, int] = TRAIN_SPLIT_SIZES,
) -> str | None:
    """檢查是否為一組完整的目前輸出；完整回傳 None，否則回傳第一個問題（含檔名）。

    第一層：完成 manifest 存在、schema／generator／seed 相符、五檔 sha256 與 manifest 一致，
    可辨識任何內容改動與中斷重建留下的新舊混合。第二層：結構檢查（見 _structure_problem）。
    """
    names = (TEST_SAMPLE_NAME, *train_sizes, CORPUS_NAME)
    return _manifest_problem(output_dir, seed, names) or _structure_problem(output_dir, sample_size, train_sizes)


def _structure_problem(output_dir: Path, sample_size: int, train_sizes: Mapping[str, int]) -> str | None:
    """逐檔解析 JSONL，檢查題目檔行數、題目／段落／corpus 鍵序、檔內 id 唯一、train 三份互斥，
    以及 corpus 的 pid 集合恰為 sample 全部段落的 pid 集合。"""
    expected = {TEST_SAMPLE_NAME: sample_size, **train_sizes}
    rows: dict[str, list[Any]] = {}
    for name in (*expected, CORPUS_NAME):
        path = output_dir / f"{name}.jsonl"
        if not path.is_file():
            return f"缺少 {path.name}"
        try:
            rows[name] = _read_jsonl_rows(path)
        except ValueError as error:
            return str(error)

    seen_train_ids: set[str] = set()
    for name, size in expected.items():
        questions = rows[name]
        if len(questions) != size:
            return f"{name}.jsonl 應有 {size} 行，實際 {len(questions)} 行"
        if not all(
            isinstance(q, dict) and list(q) == QUESTION_KEYS
            and all(isinstance(p, dict) and list(p) == PARAGRAPH_KEYS for p in q["paragraphs"])
            for q in questions
        ):
            return f"{name}.jsonl 的欄位不符合目前 schema"
        ids = {q["id"] for q in questions}
        if len(ids) != size:
            return f"{name}.jsonl 有重複 id"
        if name != TEST_SAMPLE_NAME:
            if ids & seen_train_ids:
                return f"{name}.jsonl 與其他 train 切分重疊"
            seen_train_ids |= ids

    corpus = rows[CORPUS_NAME]
    if not all(isinstance(row, dict) and list(row) == CORPUS_KEYS for row in corpus):
        return f"{CORPUS_NAME}.jsonl 的欄位不符合目前 schema"
    corpus_pids = [row["pid"] for row in corpus]
    sample_pids = {p["pid"] for q in rows[TEST_SAMPLE_NAME] for p in q["paragraphs"]}
    if len(corpus_pids) != len(set(corpus_pids)) or set(corpus_pids) != sample_pids:
        return f"{CORPUS_NAME}.jsonl 與 {TEST_SAMPLE_NAME}.jsonl 的段落不一致"
    return None


def prepare_dataset(
    dataset: str,
    splits: Mapping[str, Any],
    output_dir: Path,
    *,
    seed: int = SEED,
    sample_size: int = TEST_SAMPLE_SIZE,
    train_sizes: Mapping[str, int] = TRAIN_SPLIT_SIZES,
    logger: Any = None,
) -> dict[str, int]:
    """validation 抽測試題、train 抽互斥三份，並由測試題建 corpus；回傳各檔行數。

    先刪除舊 manifest，五檔寫完且通過結構檢查後才寫新 manifest，中斷時不會留下完成標記。
    """
    validation = splits["validation"]
    test_ids = sample_ids(validation["id"], sample_size, seed=seed)
    test_questions = _convert_selected(dataset, validation, test_ids, logger)
    train = splits["train"]
    train_questions = {
        name: _convert_selected(dataset, train, ids, logger)
        for name, ids in split_train_ids(train["id"], train_sizes, seed=seed).items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_NAME).unlink(missing_ok=True)
    counts = {TEST_SAMPLE_NAME: write_jsonl(output_dir / f"{TEST_SAMPLE_NAME}.jsonl", test_questions)}
    for name, questions in train_questions.items():
        counts[name] = write_jsonl(output_dir / f"{name}.jsonl", questions)
    counts[CORPUS_NAME] = write_jsonl(output_dir / f"{CORPUS_NAME}.jsonl", build_corpus(test_questions))

    problem = _structure_problem(output_dir, sample_size, train_sizes)
    if problem is not None:
        raise RuntimeError(f"{dataset} 輸出自我檢查失敗，不寫 {MANIFEST_NAME}：{problem}")
    spec = DATASET_SPECS[dataset]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "dataset": dataset,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "repo": spec.repo,
            "config": spec.config,
            "splits": {"validation": {"rows": len(validation)}, "train": {"rows": len(train)}},
        },
        "files": {
            f"{name}.jsonl": {"sha256": _sha256(output_dir / f"{name}.jsonl"), "line_count": count}
            for name, count in counts.items()
        },
    }
    temporary = output_dir / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, output_dir / MANIFEST_NAME)
    return counts
