from pathlib import Path

import pytest

from multihop_benchmark.settings import SettingsError, load_settings


def write_env(tmp_path: Path, text: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(text, encoding="utf-8")
    return env_file


def test_reads_roots_and_ollama_host_from_env_file(tmp_path):
    env_file = write_env(
        tmp_path,
        "PROJECT_DATA_ROOT=/data/laya_data\n"
        "PROJECT_RUNTIME_ROOT=/runtime/laya_runtime\n"
        "OLLAMA_HOST=http://127.0.0.1:12345\n",
    )

    settings = load_settings(env_file=env_file, environ={})

    assert settings.data_root == Path("/data/laya_data")
    assert settings.runtime_root == Path("/runtime/laya_runtime")
    assert settings.ollama_host == "http://127.0.0.1:12345"
    assert settings.logs_dir == Path("/data/laya_data/logs")
    assert settings.ollama_models_dir == Path("/runtime/laya_runtime/models/ollama")
    assert settings.hf_home == Path("/runtime/laya_runtime/cache/huggingface")


def test_environment_variables_override_env_file(tmp_path):
    env_file = write_env(
        tmp_path,
        "PROJECT_DATA_ROOT=/from/file\nPROJECT_RUNTIME_ROOT=/runtime/file\n",
    )

    settings = load_settings(
        env_file=env_file,
        environ={"PROJECT_DATA_ROOT": "/from/env", "OLLAMA_HOST": "http://gpu:11434"},
    )

    assert settings.data_root == Path("/from/env")
    assert settings.runtime_root == Path("/runtime/file")
    assert settings.ollama_host == "http://gpu:11434"


def test_ollama_host_defaults_to_localhost(tmp_path):
    env_file = write_env(tmp_path, "PROJECT_DATA_ROOT=/d\nPROJECT_RUNTIME_ROOT=/r\n")

    settings = load_settings(env_file=env_file, environ={})

    assert settings.ollama_host == "http://localhost:11434"


def test_missing_roots_raise_clear_error(tmp_path):
    with pytest.raises(SettingsError) as excinfo:
        load_settings(env_file=tmp_path / "missing.env", environ={})

    message = str(excinfo.value)
    assert "PROJECT_DATA_ROOT" in message
    assert "PROJECT_RUNTIME_ROOT" in message
    assert ".env.example" in message


def test_relative_root_is_rejected(tmp_path):
    env_file = write_env(tmp_path, "PROJECT_DATA_ROOT=laya_data\nPROJECT_RUNTIME_ROOT=/r\n")

    with pytest.raises(SettingsError, match="PROJECT_DATA_ROOT"):
        load_settings(env_file=env_file, environ={})
