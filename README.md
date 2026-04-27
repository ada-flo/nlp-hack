# nlp-hack

Debate-themed dialogue generation hackathon. Goal: an LSTM-based seq2seq chatbot
that produces a counter-argument given a topic and an opponent's argument.

See `Hackathon and Team Building.pdf` for the full brief.

## Quick start on a fresh server

```bash
# 1. clone
git clone git@github.com:ada-flo/nlp-hack.git && cd nlp-hack

# 2. install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. configure env (HF_TOKEN strongly recommended for faster downloads)
cp .env.example .env
$EDITOR .env

# 4. (optional) launch vLLM in a separate process / window for KO synth
bash scripts/launch_vllm.sh                 # serves on :8000

# 5. run the full pipeline
bash scripts/run_pipeline.sh
```

That's it. Final data lands at `data/processed/{train,valid,test}.jsonl`.

If `VLLM_BASE_URL` isn't set in `.env`, step 5 still produces ~15K records
(EN + KLUE-NLI Korean). With vLLM running, KO petitions synth adds ~10K more.

## Layout

```text
configs/data_sources.yaml          # caps, split ratios, source toggles
data/                              # raw/raw_manual/interim ignored; processed kept
docs/                              # plan, dataset examples
scripts/run_pipeline.sh            # one-shot pipeline runner
scripts/launch_vllm.sh             # vLLM server launcher
src/preprocess/                    # per-source adapters + merge_and_split + utils
src/synth/                         # vLLM client + counter-argument synthesis
src/model/                         # LSTM seq2seq (todo)
src/train.py                       # training entry point (todo)
```

## Roles

- **Data preparation** — see [`docs/plan_gyehun.md`](docs/plan_gyehun.md) and the
  no-AI-Hub additions in [`docs/plan_gyehun_revised.md`](docs/plan_gyehun_revised.md).
- **Model development** — `src/model/`, `src/train.py`.
- **Strategy** — technical report and slides.

## Data sources

Target: ~10K English + ~10K Korean records (caps in `configs/data_sources.yaml`).

| Lang | Source | Cap | Type | Needs |
|---|---|---|---|---|
| EN | IBM ArgQ 30K | 6,000 | real pro/con | HF |
| EN | mc-ai/conversation_dataset | 1,500 | real dialogue | HF |
| EN | Isotonic/human_assistant_conversation | 1,500 | real dialogue | HF |
| EN | SohamGhadge/casual-conversation | 1,000 | real dialogue | HF |
| KO | KLUE-NLI | 5,000 | NLI contradiction pairs | HF, no vLLM |
| KO | Korean Petitions (Korpora) | 10,000 | vLLM-synth rebuttal | **vLLM** |
| KO | K-News-Stance | best-effort | real stance pairs | manual download |
| KO | AI Hub Dialogue Summarization | best-effort | real dialogue | manual download |
| KO | AI Hub Topic Dialogue | best-effort | real dialogue | manual download |

Manual-data adapters skip cleanly when their input files are absent — the
pipeline runs to completion regardless. Drop manual data here when available:

```text
data/raw_manual/k-news-stance/k-news-stance.json
data/raw_manual/aihub/dialogue_summary/<files>.json
data/raw_manual/aihub/topic_dialogue/<files>.json
```

## Running adapters individually

`scripts/run_pipeline.sh` is the recommended path, but each adapter is also
runnable on its own:

```bash
uv run python -m src.preprocess.en_ibm_argq
uv run python -m src.preprocess.ko_klue_nli
uv run python -m src.preprocess.ko_korean_petitions   # needs vLLM
uv run python -m src.preprocess.merge_and_split
```

## vLLM setup

vLLM serves an OpenAI-compatible API, so the synthesis client treats local
and remote servers identically. Only `VLLM_BASE_URL` changes.

```bash
# On the GPU server (8x B200 example, defaults in scripts/launch_vllm.sh):
pip install vllm
bash scripts/launch_vllm.sh

# Override the model:
VLLM_MODEL=Qwen/Qwen3-72B-Instruct VLLM_TP=4 bash scripts/launch_vllm.sh

# In .env on the machine running the pipeline:
VLLM_BASE_URL=http://<gpu-host>:8000/v1
VLLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct
VLLM_API_KEY=EMPTY
```

Korean Petitions synth runs ~10K calls with concurrency 16 and chunked writes
(no progress lost on crash). With Qwen3-235B-A22B on 8 B200s expect ~15–25 min.

## Topic-leakage rule

Splits are **topic-level** for debate records (every record on a given motion
lands in exactly one of train/valid/test, never two). Sources with a uniform
placeholder topic — casual chat and KLUE-NLI — split row-wise instead via an
explicit allowlist in `src/preprocess/merge_and_split.py`. Without that
distinction, all 5K NLI rows would land in a single split.

## Troubleshooting

- **HF rate limits / slow downloads** → set `HF_TOKEN` in `.env`.
- **`SSL: CERTIFICATE_VERIFY_FAILED`** (Korpora) → `pip install --upgrade
  certifi`, then `export SSL_CERT_FILE=$(python -c 'import certifi;
  print(certifi.where())')`. Linux servers typically don't hit this; macOS
  may need `Install Certificates.command`.
- **`ko_korean_petitions.py` fails preflight** → vLLM isn't reachable at
  `VLLM_BASE_URL`. Confirm with `curl $VLLM_BASE_URL/models`.
- **Mismatched `Generating validation split` schema** (Isotonic) → already
  handled with streaming load.
