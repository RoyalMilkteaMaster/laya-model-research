# 以 `source multihop_benchmark/scripts/env.sh` 載入；uv 與 ollama 不會自己讀 .env。
#
# 從 multihop_benchmark/.env 讀 PROJECT_DATA_ROOT、PROJECT_RUNTIME_ROOT、OLLAMA_HOST
# （已匯出的同名環境變數優先），並匯出：
#   OLLAMA_MODELS、HF_HOME、OLLAMA_HOST、UV_PROJECT_ENVIRONMENT、
#   PATH（前置 ~/.local/bin 與 Runtime Root 的 applications/ollama/bin）。
# 本檔會被互動 shell source，不可使用 set -e 或 exit。

_mhb_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_mhb_env_file="$_mhb_project_dir/.env"

if [[ -f "$_mhb_env_file" ]]; then
    while IFS= read -r _mhb_line || [[ -n "$_mhb_line" ]]; do
        _mhb_line="${_mhb_line%$'\r'}"
        [[ "$_mhb_line" =~ ^[[:space:]]*(export[[:space:]]+)?(PROJECT_DATA_ROOT|PROJECT_RUNTIME_ROOT|OLLAMA_HOST)[[:space:]]*=[[:space:]]*(.*)$ ]] || continue
        _mhb_key="${BASH_REMATCH[2]}"
        _mhb_value="${BASH_REMATCH[3]}"
        _mhb_value="${_mhb_value%\"}"; _mhb_value="${_mhb_value#\"}"
        _mhb_value="${_mhb_value%\'}"; _mhb_value="${_mhb_value#\'}"
        [[ -n "${!_mhb_key:-}" ]] || export "$_mhb_key=$_mhb_value"
    done <"$_mhb_env_file"
fi

if [[ -z "${PROJECT_DATA_ROOT:-}" || -z "${PROJECT_RUNTIME_ROOT:-}" ]]; then
    echo "env.sh：缺少 PROJECT_DATA_ROOT 或 PROJECT_RUNTIME_ROOT，請建立 $_mhb_env_file（參考 .env.example）。" >&2
else
    export PROJECT_DATA_ROOT PROJECT_RUNTIME_ROOT
    export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
    export OLLAMA_MODELS="$PROJECT_RUNTIME_ROOT/models/ollama"
    export HF_HOME="$PROJECT_RUNTIME_ROOT/cache/huggingface"
    export UV_PROJECT_ENVIRONMENT="$PROJECT_RUNTIME_ROOT/environments/multihop_benchmark"
    export TORCH_DISABLE_NATIVE_JIT=1  # Laya（ModernBERT）在 WSL 無 C 編譯器，torch 2.14 需關閉 native JIT
    for _mhb_dir in "$HOME/.local/bin" "$PROJECT_RUNTIME_ROOT/applications/ollama/bin"; do
        case ":$PATH:" in
            *":$_mhb_dir:"*) ;;
            *) export PATH="$_mhb_dir:$PATH" ;;
        esac
    done
fi

unset _mhb_project_dir _mhb_env_file _mhb_line _mhb_key _mhb_value _mhb_dir
