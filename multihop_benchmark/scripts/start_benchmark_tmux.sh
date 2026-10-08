#!/usr/bin/env bash
# 用法（於任何目錄）：bash multihop_benchmark/scripts/start_benchmark_tmux.sh <run_id> [run 的其他參數，例 --limit 3]
#
# 建立 detached tmux session（預設 laya-bench-<run_id>，可用 LAYA_TMUX_SESSION 覆寫），三個視窗：
#   ollama：$OLLAMA_HOST 無回應時啟動 `ollama serve`；已有 Ollama 在執行（例如 laya-ollama session）則沿用，
#           視窗改為每 60 秒顯示一次 `ollama ps`（同一埠不能再起第二個 serve）。
#   run：等 Ollama 可連線後執行 `multihop-benchmark run --run-id <run_id> [其他參數]`（同 run_id 重啟即續跑）。
#   report：`multihop-benchmark report --run-id <run_id> --every 600`。
# 視窗內程式結束後保留 shell 以便查看輸出與退出碼。session 屬於 tmux server，關閉呼叫的終端不受影響。
# 查看：tmux attach -t <session>；結束：tmux kill-session -t <session>。
# 視窗命令交給 bash -c 再解析一次：外部值（路徑、run_id、run 參數）一律以 printf %q 轉義後才拼入，
# OLLAMA_HOST 則留給視窗內 source env.sh 後以 "$OLLAMA_HOST" 在執行期展開，不把值拼進命令文字。
set -euo pipefail

if [[ $# -lt 1 || -z "$1" || "$1" == -* ]]; then
    echo "用法：$0 <run_id> [run 的其他參數]" >&2
    exit 2
fi
run_id="$1"
shift
if [[ "$run_id" == "." || "$run_id" == ".." || "$run_id" == */* || "$run_id" == *\\* ]]; then
    echo "run_id 必須是單一路徑片段（不可為 . 或 ..，不可含 / 或 \\）：$run_id" >&2
    exit 2
fi

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "$scripts_dir/../.." && pwd)"
env_sh="$scripts_dir/env.sh"
# shellcheck source=/dev/null
source "$env_sh"
if [[ -z "${PROJECT_DATA_ROOT:-}" ]]; then
    exit 1  # env.sh 已印出原因
fi

session="${LAYA_TMUX_SESSION:-laya-bench-$run_id}"
session="${session//[.:]/_}"  # tmux session 名稱不可含 . 與 :
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "tmux session $session 已存在；請先 tmux attach -t $session 查看，或 tmux kill-session -t $session 後重試。" >&2
    exit 1
fi

prefix="source $(printf %q "$env_sh") && cd $(printf %q "$code_root")"
cli="uv run --project multihop_benchmark multihop-benchmark"
keep='echo; echo "[$(date "+%F %T")] 已結束（exit=$status）。"; exec bash'
wait_ollama='until curl -sf -o /dev/null "$OLLAMA_HOST/api/tags"; do echo "等待 Ollama $OLLAMA_HOST ..."; sleep 2; done'

if curl -sf -o /dev/null "$OLLAMA_HOST/api/tags"; then
    ollama_mode="沿用既有 Ollama"
    ollama_cmd="$prefix && echo \"沿用既有 Ollama（\$OLLAMA_HOST），不另起 ollama serve。\" && while true; do date '+%F %T'; ollama ps; sleep 60; done"
else
    ollama_mode="啟動 ollama serve"
    ollama_cmd="$prefix && ollama serve; status=\$?; $keep"
fi
run_args="$(printf ' %q' --run-id "$run_id" "$@")"
run_cmd="$prefix && $wait_ollama && $cli run$run_args; status=\$?; $keep"
report_cmd="$prefix && $cli report --run-id $(printf %q "$run_id") --every 600; status=\$?; $keep"

tmux new-session -d -s "$session" -n ollama -c "$code_root" bash -c "$ollama_cmd"
tmux new-window -t "=$session:" -n run -c "$code_root" bash -c "$run_cmd"
tmux new-window -t "=$session:" -n report -c "$code_root" bash -c "$report_cmd"

echo "已建立 tmux session $session（ollama：$ollama_mode；run：run --run-id $run_id${*:+ $*}；report：每 600 秒）"
echo "查看：tmux attach -t $session"
