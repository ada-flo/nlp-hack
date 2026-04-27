# nlp-hack

Debate-themed dialogue generation hackathon. Goal: an LSTM-based seq2seq chatbot
that produces a counter-argument given a topic and an opponent's argument.

See `Hackathon and Team Building.pdf` for the full brief.

## Roles

Per the brief, three roles:

- **Data preparation** — see [`docs/plan_gyehun.md`](docs/plan_gyehun.md).
- **Model development** — `src/model/`, `src/train.py`.
- **Strategy** — technical report and slides.

## Layout

```text
configs/                # data_sources.yaml — caps, ratios, source toggles
data/                   # raw, raw_manual, interim ignored; processed kept
src/preprocess/         # source-specific adapters + merge_and_split + utils
src/synth/              # vLLM client + counter-argument synthesis
src/model/              # LSTM seq2seq (todo)
src/train.py            # training entry point (todo)
```

## Setup

```bash
uv sync
```

## Data pipeline

Target: ~10K English + ~10K Korean records.

| Lang | Source | Cap | Type | Needs |
|---|---|---|---|---|
| EN | IBM ArgQ 30K | 6,000 | real pro/con | HF only |
| EN | mc-ai/conversation_dataset | 1,500 | real dialogue | HF only |
| EN | Isotonic/human_assistant_conversation | 1,500 | real dialogue | HF only |
| EN | SohamGhadge/casual-conversation | 1,000 | real dialogue | HF only |
| KO | KLUE-NLI (HF) | 5,000 | NLI contradiction pairs | HF only (no vLLM) |
| KO | Korean Petitions (Korpora) | 10,000 | vLLM-synth rebuttal | **vLLM running** |
| KO | K-News-Stance | best-effort | real stance pairs | manual download |
| KO | AI Hub Dialogue Summarization | best-effort | real dialogue | manual download |
| KO | AI Hub Topic Dialogue | best-effort | real dialogue | manual download |

### Run order

```bash
# English (no external services needed)
uv run python -m src.preprocess.en_ibm_argq
uv run python -m src.preprocess.en_mc_conversation
uv run python -m src.preprocess.en_isotonic_conversation
uv run python -m src.preprocess.en_casual_conversation

# Korean — runs without vLLM, fast Korean baseline
uv run python -m src.preprocess.ko_klue_nli

# Korean — start vLLM first (see "Synthetic data via vLLM" below)
uv run python -m src.preprocess.ko_korean_petitions

# Korean — these skip cleanly if manual data isn't present
uv run python -m src.preprocess.ko_k_news_stance
uv run python -m src.preprocess.ko_aihub_dialogue_summary
uv run python -m src.preprocess.ko_aihub_topic_dialogue

# Merge per-source JSONLs into final train/valid/test (80/10/10)
uv run python -m src.preprocess.merge_and_split
```

Outputs land at `data/processed/{train,valid,test}.jsonl`.

### Manual data drop locations

When the manually-downloaded corpora become available:

```text
data/raw_manual/k-news-stance/k-news-stance.json
data/raw_manual/aihub/dialogue_summary/<files>.json
data/raw_manual/aihub/topic_dialogue/<files>.json
```

The corresponding adapters detect the path and run automatically.

## Synthetic data via vLLM

vLLM serves an OpenAI-compatible API, so the same client code targets either a
local or a remote server — only the URL changes.

```bash
# Launch vLLM (local example)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# In another shell:
cp .env.example .env       # edit VLLM_BASE_URL if remote
export $(grep -v '^#' .env | xargs)
uv run python -m src.synth.counterargument   # one-call smoke test
uv run python -m src.preprocess.ko_korean_petitions  # full 10K run, ~20 min
```

For a remote server, set `VLLM_BASE_URL=https://your-host/v1` in `.env` and the
same code runs unchanged.

## Topic-leakage rule

Splits are **topic-level** for debate records (every record on motion *X* lands
in one split — train, valid, *or* test, never two). Casual-conversation
records share a placeholder topic so they're split row-wise to spread fluency
examples across all three. Implemented in `src/preprocess/merge_and_split.py`.
