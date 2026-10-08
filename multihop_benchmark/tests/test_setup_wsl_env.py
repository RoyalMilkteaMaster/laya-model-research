"""setup_wsl_env.sh 的 Ollama 雜湊驗證必須 fail closed（以 stub curl／uv 與暫存根目錄執行）。"""

import hashlib
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
ARCHIVE_BYTES = b"not really a zstd tarball"
ASSET = "ollama-linux-amd64.tar.zst"

CURL_STUB = """#!/usr/bin/env bash
# 依 STUB_MANIFEST 模擬 GitHub release：tarball 一律成功，sha256sum.txt 依模式回應。
out=""; url=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o) out="$2"; shift 2 ;;
        -C|--retry) shift 2 ;;
        -*) shift ;;
        *) url="$1"; shift ;;
    esac
done
echo "$url" >> "$STUB_LOG"
if [[ "$url" == */sha256sum.txt ]]; then
    case "$STUB_MANIFEST" in
        fail) exit 22 ;;
        missing) printf '%s  ./ollama-darwin.tgz\\n' "$(printf 0%.0s {1..64})" > "$out" ;;
        mismatch) printf '%s  ./%s\\n' "$(printf f%.0s {1..64})" "$STUB_ASSET" > "$out" ;;
        match) printf '%s  ./%s\\n' "$STUB_DIGEST" "$STUB_ASSET" > "$out" ;;
    esac
else
    printf '%s' "$STUB_ARCHIVE_BYTES" > "$out"
fi
"""

UV_STUB = """#!/usr/bin/env bash
# 安裝與 sync 都不做事；解壓 fallback（uv run）被呼叫時留下記號。
echo "uv $*" >> "$STUB_LOG"
[[ "$1" == "--version" ]] && echo "uv 0.0.0-stub"
[[ "$1" == "run" ]] && touch "$STUB_EXTRACT_MARKER"
exit 0
"""


def write_stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def run_setup(tmp_path: Path, manifest_mode: str) -> tuple[subprocess.CompletedProcess, dict[str, Path]]:
    home = tmp_path / "home"
    stub_bin = home / ".local" / "bin"
    stub_bin.mkdir(parents=True)
    write_stub(stub_bin, "curl", CURL_STUB)
    write_stub(stub_bin, "uv", UV_STUB)
    paths = {
        "runtime": tmp_path / "runtime",
        "log": tmp_path / "stub.log",
        "extract_marker": tmp_path / "extract-called",
    }
    env = {
        "HOME": str(home),
        "PATH": f"{stub_bin}:/usr/bin:/bin",
        "PROJECT_DATA_ROOT": str(tmp_path / "data"),
        "PROJECT_RUNTIME_ROOT": str(paths["runtime"]),
        "OLLAMA_HOST": "http://127.0.0.1:9",
        "STUB_MANIFEST": manifest_mode,
        "STUB_ASSET": ASSET,
        "STUB_ARCHIVE_BYTES": ARCHIVE_BYTES.decode(),
        "STUB_DIGEST": hashlib.sha256(ARCHIVE_BYTES).hexdigest(),
        "STUB_LOG": str(paths["log"]),
        "STUB_EXTRACT_MARKER": str(paths["extract_marker"]),
    }
    # 複製腳本到暫存專案目錄，避免在真實專案產生 .env。
    project_scripts = tmp_path / "project" / "scripts"
    project_scripts.mkdir(parents=True)
    for name in ("setup_wsl_env.sh", "env.sh"):
        (project_scripts / name).write_bytes((SCRIPTS_DIR / name).read_bytes())
    result = subprocess.run(
        ["bash", str(project_scripts / "setup_wsl_env.sh")],
        env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )
    return result, paths


@pytest.mark.parametrize(
    ("manifest_mode", "reason"),
    [
        ("fail", "sha256sum.txt"),
        ("missing", ASSET),
        ("mismatch", "sha256 不符"),
    ],
)
def test_unverified_ollama_archive_is_never_extracted(tmp_path, manifest_mode, reason):
    result, paths = run_setup(tmp_path, manifest_mode)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert reason in output
    applications = paths["runtime"] / "applications"
    assert not (applications / "ollama").exists()
    assert not (applications / "ollama.partial").exists()
    assert not paths["extract_marker"].exists()


def test_mismatched_archive_is_deleted_so_rerun_downloads_again(tmp_path):
    _, paths = run_setup(tmp_path, "mismatch")

    downloads = paths["runtime"] / "cache" / "downloads"
    assert not list(downloads.glob(f"*{ASSET}*"))


def test_verified_archive_proceeds_to_extraction(tmp_path):
    # 對照組：證明上面的 fail closed 不是 stub 環境本身造成——雜湊相符時會進入解壓
    # （stub tarball 不是真的 zstd，tar 必定失敗而走 uv run fallback）。
    result, paths = run_setup(tmp_path, "match")

    assert "sha256 已驗證" in result.stdout, result.stdout + result.stderr
    assert paths["extract_marker"].exists()
