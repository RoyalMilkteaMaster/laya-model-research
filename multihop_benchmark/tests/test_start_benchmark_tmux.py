"""start_benchmark_tmux.sh：三視窗命令、沿用或啟動 Ollama、session 已存在時拒絕（以 stub tmux／curl 執行）。"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_benchmark_tmux.sh"

TMUX_STUB = """#!/usr/bin/env bash
# 每次呼叫以 NUL 分隔記下 argv；has-session 依 STUB_HAS_SESSION 回應。
printf '%s\\0' "$@" >> "$STUB_LOG"; printf '\\n' >> "$STUB_LOG"
[[ "$1" == "has-session" ]] && exit "${STUB_HAS_SESSION:-1}"
exit 0
"""
CURL_STUB = """#!/usr/bin/env bash
exit "${STUB_CURL_EXIT:-0}"
"""


@pytest.fixture
def stub_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("tmux", TMUX_STUB), ("curl", CURL_STUB)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(tmp_path / "tmux.log"),
        "PROJECT_DATA_ROOT": str(tmp_path / "data"),
        "PROJECT_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "OLLAMA_HOST": "http://localhost:11434",
    }
    env.pop("LAYA_TMUX_SESSION", None)
    return tmp_path, env


def tmux_calls(tmp_path):
    raw = (tmp_path / "tmux.log").read_bytes().split(b"\n")
    return [[part.decode() for part in call.split(b"\0") if part] for call in raw if call]


def window_commands(calls):
    return {call[call.index("-n") + 1]: call[-1] for call in calls if call[0] in ("new-session", "new-window")}


def run_script(env, *args, **overrides):
    return subprocess.run(["bash", str(SCRIPT), *args], env={**env, **overrides}, capture_output=True, text=True)


def test_creates_three_windows_reusing_running_ollama(stub_env):
    tmp_path, env = stub_env
    result = run_script(env, "smoke", "--limit", "3")
    assert result.returncode == 0, result.stderr

    calls = tmux_calls(tmp_path)
    new_session = next(call for call in calls if call[0] == "new-session")
    assert new_session[new_session.index("-s") + 1] == "laya-bench-smoke" and "-d" in new_session
    commands = window_commands(calls)
    assert list(commands) == ["ollama", "run", "report"]
    assert "&& ollama serve;" not in commands["ollama"] and "ollama ps" in commands["ollama"]
    assert "multihop-benchmark run --run-id smoke --limit 3" in commands["run"]
    assert "multihop-benchmark report --run-id smoke --every 600" in commands["report"]
    for command in commands.values():
        assert "scripts/env.sh" in command


def test_starts_ollama_serve_when_not_running(stub_env):
    tmp_path, env = stub_env
    result = run_script(env, "r1", STUB_CURL_EXIT="7")
    assert result.returncode == 0, result.stderr
    commands = window_commands(tmux_calls(tmp_path))
    assert "&& ollama serve;" in commands["ollama"]
    assert "until curl" in commands["run"]  # run 視窗等 Ollama 就緒


def test_refuses_existing_session(stub_env):
    tmp_path, env = stub_env
    result = run_script(env, "smoke", STUB_HAS_SESSION="0")
    assert result.returncode == 1
    assert "已存在" in result.stderr
    assert not [call for call in tmux_calls(tmp_path) if call[0] in ("new-session", "new-window")]


def test_session_name_override_and_sanitized(stub_env):
    tmp_path, env = stub_env
    assert run_script(env, "a.b", LAYA_TMUX_SESSION="bench:x.y").returncode == 0
    new_session = next(call for call in tmux_calls(tmp_path) if call[0] == "new-session")
    assert new_session[new_session.index("-s") + 1] == "bench_x_y"


def test_requires_run_id(stub_env):
    _, env = stub_env
    assert run_script(env).returncode == 2


# ---- 實際執行產生的視窗命令：外部值只能成為資料，不能成為 shell 程式碼（Review r1 Finding）----

EXEC_TMUX_STUB = """#!/usr/bin/env bash
# has-session 回「不存在」；new-session／new-window 以 bash -c 實際執行最後一個參數（視窗命令）。
[[ "$1" == "has-session" ]] && exit 1
if [[ "$1" == "new-session" || "$1" == "new-window" ]]; then
    timeout 10 bash -c "${@: -1}" </dev/null >>"$STUB_DIR/window.out" 2>&1
fi
exit 0
"""
ARGV_STUB = """#!/usr/bin/env bash
# 以 NUL 分隔記下 argv；`ollama ps` 結束呼叫它的沿用迴圈。
printf '%s\\0' "$(basename "$0")" "$@" >> "$STUB_DIR/argv.log"; printf '\\n' >> "$STUB_DIR/argv.log"
[[ "$(basename "$0")" == "ollama" && "$1" == "ps" ]] && kill "$PPID"
exit 0
"""
FIRST_CALL_CURL_STUB = """#!/usr/bin/env bash
# 第一次呼叫（腳本判斷 Ollama 是否在跑）回 STUB_CURL_EXIT，之後（run 視窗等待就緒）一律成功。
if [[ ! -e "$STUB_DIR/curl.called" ]]; then touch "$STUB_DIR/curl.called"; exit "${STUB_CURL_EXIT:-0}"; fi
exit 0
"""
PAYLOADS = ("'; touch {marker}; echo '", "two words", "x; touch {marker}", "$(touch {marker})", "`touch {marker}`")


@pytest.fixture
def exec_env(tmp_path):
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    home.mkdir()
    for name, body in (("tmux", EXEC_TMUX_STUB), ("uv", ARGV_STUB), ("ollama", ARGV_STUB), ("curl", FIRST_CALL_CURL_STUB)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(home),  # env.sh 只前置 $HOME/.local/bin，避免真的 uv 蓋過 stub
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_DIR": str(tmp_path),
        "PROJECT_DATA_ROOT": str(tmp_path / "data"),
        "PROJECT_RUNTIME_ROOT": str(tmp_path / "runtime"),
    }
    env.pop("LAYA_TMUX_SESSION", None)
    return tmp_path, env


def executed_argv(tmp_path):
    raw = (tmp_path / "argv.log").read_bytes().split(b"\n")
    return [[part.decode() for part in call.split(b"\0") if part] for call in raw if call]


@pytest.mark.parametrize("curl_exit", ["0", "7"])  # 沿用既有 Ollama／啟動 ollama serve 兩條分支
def test_generated_window_commands_treat_metacharacters_as_data(exec_env, curl_exit):
    tmp_path, env = exec_env
    marker = tmp_path / "INJECTED"
    payloads = [p.format(marker=marker) for p in PAYLOADS]
    run_id = """r '; cd "$STUB_DIR"; touch INJECTED; echo ';$(cd "$STUB_DIR"; touch INJECTED)"""  # 不含 /
    host = "http://h" + "".join(payloads)
    result = run_script(env, run_id, "--models", *payloads, OLLAMA_HOST=host, STUB_CURL_EXIT=curl_exit)
    assert result.returncode == 0, result.stderr

    assert not marker.exists(), (tmp_path / "window.out").read_text()
    calls = executed_argv(tmp_path)
    uv_calls = [call[1:] for call in calls if call[0] == "uv"]
    cli = ["run", "--project", "multihop_benchmark", "multihop-benchmark"]
    assert [cli + ["run", "--run-id", run_id, "--models", *payloads]] == [c for c in uv_calls if c[4] == "run"]
    assert [cli + ["report", "--run-id", run_id, "--every", "600"]] == [c for c in uv_calls if c[4] == "report"]
    expected_ollama = ["ps"] if curl_exit == "0" else ["serve"]
    assert [call[1:] for call in calls if call[0] == "ollama"] == [expected_ollama]
    if curl_exit == "0":  # 沿用訊息顯示的是字面上的 host
        assert host in (tmp_path / "window.out").read_text()


@pytest.mark.parametrize("run_id", ["..", ".", "a/b", "/abs", "a\\b"])
def test_rejects_run_id_that_is_not_a_single_path_segment(stub_env, run_id):
    tmp_path, env = stub_env
    result = run_script(env, run_id)
    assert result.returncode == 2
    assert not (tmp_path / "tmux.log").exists() or not [
        call for call in tmux_calls(tmp_path) if call[0] in ("new-session", "new-window")]
