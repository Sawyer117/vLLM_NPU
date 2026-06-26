#!/usr/bin/env bash
# ============================================================================
#  DSV4 serve - NODE 0 (MASTER) on 80.5.5.136
#  Self-contained + pre-flight self-checks. Uses the FULL PATH to the
#  vllm-dflash python/vllm so it does NOT matter which conda env is active.
#  Run:   bash test_bench/serve_node0_136.sh
# ============================================================================

# ---- fixed locations (edit ONLY if your paths differ) ----------------------
REPO=/home/f00518697/m84379596/DFlash/vLLM_NPU
ENVDIR=/home/f00518697/m84379596/conda/vllm-dflash
VLLM="$ENVDIR/bin/vllm"
PYBIN="$ENVDIR/bin/python"
MODEL=/home/f00518697/m84379596/Huggingface/DeepSeek-V4-Flash-bf16
DSA="$REPO/vllm-ascend/vllm_ascend/attention/dsa_v1.py"

# ---- node-specific ---------------------------------------------------------
NODE_RANK=0
MASTER_ADDR=80.5.5.136
MASTER_PORT=29501
SERVE_PORT=30000

# ---- CANN (NODE 0 / 136 uses the combined env script) ----------------------
echo "[cann] sourcing /home/a00652497/900env_npu.sh"
source /home/a00652497/900env_npu.sh

# ===========================================================================
#  PRE-FLIGHT SELF-CHECKS  (abort on any failure)
# ===========================================================================
fail(){ echo "FATAL: $*" >&2; exit 1; }
echo "==================== PRE-FLIGHT ($(hostname), node-rank $NODE_RANK) ===================="

[ -x "$VLLM" ]            || fail "vllm not found / not executable: $VLLM"
[ -x "$PYBIN" ]          || fail "python not found: $PYBIN"
[ -f "$MODEL/config.json" ] || fail "model config.json missing under $MODEL"
[ -f "$DSA" ]            || fail "dsa_v1.py missing: $DSA"

# patched source on disk?
FIXN=$(grep -c "Moh_7596-fix" "$DSA" 2>/dev/null || echo 0)
MARK=$(grep -c "MOH-PATCH"   "$DSA" 2>/dev/null || echo 0)
echo "[check] dsa_v1.py  Moh_7596-fix=$FIXN  MOH-PATCH=$MARK"
[ "$FIXN" -ge 4 ] || fail "dsa_v1.py is NOT patched (Moh_7596-fix=$FIXN). Pull the fork branch file first."
[ "$MARK" -ge 1 ] || fail "dsa_v1.py missing load-marker (MOH-PATCH=$MARK). Pull the latest fork branch file."

# does the runtime (this python) actually import vllm_ascend from THIS checkout?
EDLOC=$("$PYBIN" -m pip show vllm-ascend 2>/dev/null | awk -F': ' '/Editable project location/{print $2}')
echo "[check] vllm-ascend editable -> ${EDLOC:-<none>}"
[ "$EDLOC" = "$REPO/vllm-ascend" ] || fail "vllm-ascend editable is '$EDLOC', expected '$REPO/vllm-ascend'. Wrong env / not editable-installed."

# kill stale servers, then VERIFY they are gone
echo "[kill] removing any stale vllm processes ..."
pkill -9 -f "vllm serve|EngineCore|multiproc_executor|VLLM::" 2>/dev/null || true
sleep 5
if ps -ef | grep -iE 'vllm serve|EngineCore|VLLM::|multiproc_executor' | grep -v grep ; then
  fail "stale vllm processes still alive after kill (see above)."
fi
# master: serve port must be free
if curl --noproxy '*' -s -m 3 "http://$MASTER_ADDR:$SERVE_PORT/v1/models" >/dev/null 2>&1 ; then
  echo "[port] something still answering on :$SERVE_PORT -> $(ss -ltnp 2>/dev/null | grep ":$SERVE_PORT" || true)"
  fail "port $SERVE_PORT still in use by another server. Kill it first (ss -ltnp | grep $SERVE_PORT)."
fi
echo "==================== CHECKS PASSED -> launching ===================="

# ===========================================================================
#  RUNTIME ENV
# ===========================================================================
export PYTHONPATH="$REPO/vllm:$REPO/vllm-ascend:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export DSV4_VLLM_SERVE_PATCH=1
unset  DFLASH_DISABLE_QLI                  2>/dev/null || true
unset  VLLM_ASCEND_ENABLE_FLASHCOMM1       2>/dev/null || true
export MASTER_ADDR MASTER_PORT
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | awk '/80\.5\.5\./{print $2; exit}')
export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export GLOO_USE_IPV6=0
echo "[env] ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES  IFNAME=$GLOO_SOCKET_IFNAME  DFLASH_DISABLE_QLI=${DFLASH_DISABLE_QLI:-<unset>}"

# ===========================================================================
#  LAUNCH  (full-path vllm-dflash binary; watch for the [MOH-PATCH] line)
# ===========================================================================
echo "[run] $VLLM serve  (node-rank $NODE_RANK, TP=16, master $MASTER_ADDR:$MASTER_PORT)"
exec "$VLLM" serve "$MODEL" \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 --max-model-len 1024 --max-num-batched-tokens 1024 \
  --block-size 128 --no-enable-prefix-caching \
  --enforce-eager \
  --host 0.0.0.0 --port "$SERVE_PORT"
