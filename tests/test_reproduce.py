"""Publication checks must reject altered data, not only produce plausible tables."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("publication_reproduce", Path(__file__).resolve().parents[1] / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


@pytest.fixture(scope="module")
def frozen_run():
    records, skipped = reproduce.agg.load_records(reproduce.RUN / "records")
    assert skipped == 0
    return records, reproduce.agg.summarize(records)


def test_answer_edit_is_detected(frozen_run):
    records, rows = frozen_run
    changed = copy.deepcopy(records)
    record = next(r for r in changed if r["status"] == "done" and r["em"] == 1)
    record["prediction"] = "intentionally incorrect answer for verification"
    with pytest.raises(ValueError, match="Answer score mismatch"):
        reproduce.verify_records(changed, rows)


def test_missing_question_is_detected(frozen_run):
    records, rows = frozen_run
    with pytest.raises(ValueError, match="Expected 12,000"):
        reproduce.verify_records(records[:-1], rows)


def test_changed_file_fails_checksum(tmp_path, monkeypatch):
    (tmp_path / "provenance").mkdir()
    (tmp_path / "sample.json").write_bytes(b"modified")
    digest = hashlib.sha256(b"original").hexdigest()
    (tmp_path / "provenance/SHA256SUMS").write_text(f"{digest}  sample.json\n")
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        reproduce.verify_checksums()


def test_output_cannot_overwrite_archived_data():
    with pytest.raises(ValueError, match="frozen inputs"):
        reproduce.main(["--output", str(reproduce.DATA)])
