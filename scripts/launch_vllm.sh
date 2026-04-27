#!/usr/bin/env bash
# Launch vLLM serving the model used for Korean rebuttal synthesis.
#
# Prerequisites:
#   - vLLM installed in your GPU environment: `pip install vllm`
#   - GPU server with enough memory for the chosen model
#
# Defaults assume an 8x B200 setup. Override via env vars:
#   VLLM_MODEL  — HF model id (default Qwen/Qwen3-235B-A22B-Instruct)
#   VLLM_TP     — tensor-parallel size (default 8)
#   VLLM_PORT   — server port (default 8000)
#   VLLM_MAX_LEN — max context length (default 4096)

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-235B-A22B-Instruct}"
TP="${VLLM_TP:-8}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_LEN:-4096}"

echo "Launching vLLM"
echo "  model:           $MODEL"
echo "  tensor-parallel: $TP"
echo "  port:            $PORT"
echo "  max-model-len:   $MAX_LEN"
echo

EXTRA_FLAGS=()
case "$MODEL" in
  *A22B*|*A3B*|*MoE*)
    EXTRA_FLAGS+=(--enable-expert-parallel)
    ;;
esac
case "$MODEL" in
  *EXAONE*)
    EXTRA_FLAGS+=(--trust-remote-code)
    ;;
esac

exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  "${EXTRA_FLAGS[@]}"
