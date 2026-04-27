#!/usr/bin/env bash
# Run the full data generation pipeline.
#
# Prerequisites:
#   1. `uv` installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   2. .env created (copy from .env.example) with HF_TOKEN and optionally VLLM_*
#   3. (optional, for Korean Petitions synth) vLLM running — see scripts/launch_vllm.sh
#
# Adapters whose data isn't available will skip cleanly. The merge step picks
# up whatever data/interim/*.jsonl files exist.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "=== [1/4] Sync uv environment ==="
uv sync

echo
echo "=== [2/4] EN adapters (no vLLM, no manual data) ==="
uv run python -m src.preprocess.en_ibm_argq
uv run python -m src.preprocess.en_mc_conversation
uv run python -m src.preprocess.en_isotonic_conversation
uv run python -m src.preprocess.en_casual_conversation

echo
echo "=== [3/4] KO adapters ==="
if [ -n "${VLLM_BASE_URL:-}" ]; then
  echo "--- Topic-seeded debate synth (VLLM_BASE_URL=$VLLM_BASE_URL) ---"
  uv run python -m src.preprocess.ko_debate_synth
  echo "--- Korean Petitions synth ---"
  uv run python -m src.preprocess.ko_korean_petitions
else
  echo "--- KO synth steps: VLLM_BASE_URL not set, skipping ---"
fi

echo "--- Manual-data-gated KO adapters (skip cleanly if data absent) ---"
uv run python -m src.preprocess.ko_k_news_stance
uv run python -m src.preprocess.ko_aihub_dialogue_summary
uv run python -m src.preprocess.ko_aihub_topic_dialogue

echo
echo "=== [4/4] Merge interim files into train/valid/test ==="
uv run python -m src.preprocess.merge_and_split

echo
echo "=== DONE ==="
echo "Final splits in data/processed/:"
wc -l data/processed/*.jsonl
