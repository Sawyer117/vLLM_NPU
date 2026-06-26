#!/usr/bin/env bash
# ============================================================================
#  verify.sh — one-call verification of the 2-node DSV4 serving:
#    1) reads BOTH nodes' actual running `vllm serve` command (local + peer via
#       ssh best-effort) and flags if it is NOT the vllm-dflash full-path binary
#    2) HTTP health check
#    3) functional scan (prefill-vs-decode A sweep, B/D prompts, E determinism)
#  Run:  bash test_bench/verify.sh
#  If ssh to the peer is not set up, just run this same script on the other node.
# ============================================================================
HOST=http://80.5.5.136:30000
MODEL=/home/f00518697/m84379596/Huggingface/DeepSeek-V4-Flash-bf16
NODES=(80.5.5.136 80.5.5.60)
EXPECT_VLLM="/conda/vllm-dflash/bin/vllm"

# ---- per-node report (also shipped over ssh to the peer) -------------------
node_report(){
  echo "  host: $(hostname)"
  local line
  line=$(ps -eo pid,lstart,cmd 2>/dev/null | grep -E '[v]llm serve' | head -1)
  if [ -z "$line" ]; then echo "  PROC: <no 'vllm serve' process running!>"; return; fi
  echo "  PROC: $line"
  case "$line" in
    */conda/vllm-dflash/bin/vllm*) echo "  ENV : OK  (vllm-dflash full-path)";;
    *speculator*)                  echo "  ENV : !!! WRONG -> speculator env, NOT vllm-dflash";;
    *)                             echo "  ENV : ??  not the expected vllm-dflash full-path binary";;
  esac
  echo "  RANK: $(echo "$line" | grep -oE 'node-rank [0-9]+')"
  local log; log=$(ls -t "$HOME"/serve*.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    echo "  LOG : $log   [MOH-PATCH] lines = $(grep -c 'MOH-PATCH' "$log" 2>/dev/null)"
  else
    echo "  LOG : <no ~/serve*.log found (start with '... | tee ~/serveN.log' to capture)>"
  fi
}

echo "################ 1) 两节点 进程 / 环境 / 补丁 ################"
SELF=$(ip -o -4 addr show 2>/dev/null | awk '/80\.5\.5\./{print $4}' | cut -d/ -f1 | head -1)
for n in "${NODES[@]}"; do
  echo "===== node $n ====="
  if [ "$n" = "$SELF" ]; then
    node_report
  else
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$n" "$(declare -f node_report); node_report" 2>/dev/null \
      || echo "  (无法 ssh 到 $n — 请在 $n 上单独跑 bash test_bench/verify.sh 看它本地的报告)"
  fi
done

echo; echo "################ 2) HTTP 健康检查 ################"
if ! curl --noproxy '*' -s -m 5 "$HOST/v1/models" \
     | python3 -c 'import json,sys;print("  models:",[m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null ; then
  echo "  !!! $HOST 无响应 — server 没起来或端口不对"; exit 1
fi

# ---- helper ----------------------------------------------------------------
ask(){ # $1=prompt $2=max_tokens
  local resp
  resp=$(curl --noproxy '*' -s "$HOST/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 - "$1" "$2" "$MODEL" <<'PY'
import json,sys
print(json.dumps({"model":sys.argv[3],"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0}))
PY
)")
  echo "$resp" | python3 -c 'import json,sys
d=json.load(sys.stdin)
try:
  c=d["choices"][0]; print("   text:",repr(c["message"]["content"]),"| finish:",c.get("finish_reason"))
except Exception:
  print("   RAW:",json.dumps(d)[:400])'
}

echo; echo "################ 3) A. prefill vs decode 扫描 ################"
for N in 1 2 4 8 16 32; do echo "[max_tokens=$N]"; ask "What is 2+2? Answer only the number." "$N"; done

echo; echo "################ 4) B/D 其它 prompt ################"
ask "What is the capital of France?" 20
ask "Count from 1 to 10." 20
echo "[raw completion: 'The capital of France is']"
curl --noproxy '*' -s "$HOST/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print("   text:",repr(d["choices"][0]["text"]),"| finish:",d["choices"][0].get("finish_reason"))' 2>/dev/null

echo; echo "################ 5) E. 确定性（两次应完全一致）################"
ask "What is 2+2?" 16
ask "What is 2+2?" 16
echo; echo "==== verify done ===="
