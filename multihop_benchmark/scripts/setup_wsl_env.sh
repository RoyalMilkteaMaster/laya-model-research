#!/usr/bin/env bash
# 在 WSL 使用者空間安裝 multihop_benchmark 的執行環境（不需 sudo、可重複執行）。
#
# 步驟：uv → Python 3.12 → 三根目錄 → .env → venv（uv sync）→ Ollama 官方 tarball。
# 已存在的安裝物不重新下載。本腳本不拉取任何模型。
#
# 可覆寫的環境變數：
#   PROJECT_DATA_ROOT / PROJECT_RUNTIME_ROOT / OLLAMA_HOST  產生 .env 時的預設值
#   OLLAMA_VERSION                                          Ollama release tag（預設 v0.34.4）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_ROOT="$(dirname "$PROJECT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.34.4}"
OLLAMA_ASSET="ollama-linux-amd64.tar.zst"
PYTHON_VERSION="3.12"

log() { printf '[setup] %s\n' "$*"; }

if [[ $EUID -eq 0 ]]; then
    echo "請以一般使用者執行，本腳本不需要也不應使用 root。" >&2
    exit 1
fi

# ---- 1. uv ----
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
    log "uv 已存在：$(uv --version)"
else
    log "安裝 uv 到 ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
    log "uv 安裝完成：$(uv --version)"
fi

# ---- 2. .env 與三根目錄 ----
if [[ -f "$ENV_FILE" ]]; then
    log ".env 已存在，沿用：$ENV_FILE"
else
    log "產生 .env：$ENV_FILE"
    cat >"$ENV_FILE" <<EOF
PROJECT_DATA_ROOT=${PROJECT_DATA_ROOT:-$(dirname "$CODE_ROOT")/laya_data}
PROJECT_RUNTIME_ROOT=${PROJECT_RUNTIME_ROOT:-$HOME/laya_runtime}
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
EOF
fi

# 以 env.sh 讀 .env 並匯出 OLLAMA_MODELS、HF_HOME、UV_PROJECT_ENVIRONMENT 等變數。
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

log "Data Root：$PROJECT_DATA_ROOT"
log "Runtime Root：$PROJECT_RUNTIME_ROOT"
mkdir -p \
    "$PROJECT_DATA_ROOT"/{datasets,indexes,laya_decisions,runs,reports,logs} \
    "$PROJECT_RUNTIME_ROOT"/applications \
    "$PROJECT_RUNTIME_ROOT"/models/{ollama,laya_multihop} \
    "$PROJECT_RUNTIME_ROOT"/cache/{huggingface,downloads} \
    "$PROJECT_RUNTIME_ROOT"/environments

# ---- 3. Python 3.12 與 venv ----
log "安裝 Python $PYTHON_VERSION（uv 管理）"
uv python install "$PYTHON_VERSION"
log "建立 venv：$UV_PROJECT_ENVIRONMENT"
uv sync --locked --project "$PROJECT_DIR" --python "$PYTHON_VERSION"

# ---- 4. Ollama ----
OLLAMA_DIR="$PROJECT_RUNTIME_ROOT/applications/ollama"
if [[ -x "$OLLAMA_DIR/bin/ollama" ]] && "$OLLAMA_DIR/bin/ollama" --version >/dev/null 2>&1; then
    log "Ollama 已存在：$OLLAMA_DIR"
else
    DOWNLOAD_DIR="$PROJECT_RUNTIME_ROOT/cache/downloads"
    ARCHIVE="$DOWNLOAD_DIR/ollama-$OLLAMA_VERSION-$OLLAMA_ASSET"
    BASE_URL="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION"
    MANIFEST="$DOWNLOAD_DIR/ollama-$OLLAMA_VERSION-sha256sum.txt"

    # 雜湊驗證 fail closed：取不到 sha256sum.txt、未列此檔或不符時一律中止，不解壓、不執行。
    if ! curl -fsSL --retry 3 -o "$MANIFEST.part" "$BASE_URL/sha256sum.txt"; then
        rm -f "$MANIFEST.part"
        echo "無法下載 $BASE_URL/sha256sum.txt，無法驗證 Ollama tarball，中止安裝（已下載的 tarball 保留供續傳）。" >&2
        exit 1
    fi
    mv "$MANIFEST.part" "$MANIFEST"
    expected="$(awk -v f="$OLLAMA_ASSET" '$2 == f || $2 == "./"f || $2 == "*"f {print $1}' "$MANIFEST")"
    if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
        echo "sha256sum.txt 未列出 $OLLAMA_ASSET 的有效 sha256，無法驗證，中止安裝。" >&2
        exit 1
    fi

    if [[ -f "$ARCHIVE" ]]; then
        log "沿用已下載的 tarball：$ARCHIVE"
    else
        log "下載 Ollama $OLLAMA_VERSION：$BASE_URL/$OLLAMA_ASSET"
        curl -fL --retry 3 -C - -o "$ARCHIVE.part" "$BASE_URL/$OLLAMA_ASSET"
        mv "$ARCHIVE.part" "$ARCHIVE"
    fi

    actual="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    if [[ "$expected" != "$actual" ]]; then
        rm -f "$ARCHIVE"
        echo "Ollama tarball sha256 不符（預期 $expected，實際 $actual），已刪除，請重跑。" >&2
        exit 1
    fi
    log "sha256 已驗證：$actual"

    STAGING="$OLLAMA_DIR.partial"
    rm -rf "$STAGING"
    mkdir -p "$STAGING"
    if tar --zstd -xf "$ARCHIVE" -C "$STAGING" 2>/dev/null; then
        log "以 tar --zstd 解壓"
    else
        # 系統沒有 zstd 程式：改用 uv 管理的 Python + zstandard 串流解壓（不動專案依賴）。
        log "tar --zstd 不可用，改用 Python zstandard 解壓"
        rm -rf "$STAGING"
        mkdir -p "$STAGING"
        uv run --no-project --python "$PYTHON_VERSION" --with zstandard python - "$ARCHIVE" "$STAGING" <<'PY'
import sys
import tarfile

import zstandard

archive, dest = sys.argv[1], sys.argv[2]
with open(archive, "rb") as raw:
    with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(dest, filter="tar")
PY
    fi
    rm -rf "$OLLAMA_DIR"
    mv "$STAGING" "$OLLAMA_DIR"
    log "Ollama 已安裝：$("$OLLAMA_DIR/bin/ollama" --version 2>&1 | tail -n 1 | sed 's/^Warning: //')"
fi

cat <<EOF

[setup] 完成。後續步驟：
  source $SCRIPT_DIR/env.sh
  tmux new-session -d -s laya-ollama "bash -c 'source $SCRIPT_DIR/env.sh && exec ollama serve'"
  curl -s -o /dev/null -w '%{http_code}\n' \$OLLAMA_HOST/api/tags
  uv run --project $PROJECT_DIR multihop-benchmark setup-check
EOF
