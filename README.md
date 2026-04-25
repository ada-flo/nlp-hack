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
configs/                # data_sources.yaml and other configs
data/                   # raw, raw_manual, interim ignored by Git; processed kept
src/preprocess/         # source-specific adapters + common.py + merge_and_split.py
src/model/              # LSTM seq2seq
src/train.py            # training entry point
```

## Setup

```bash
uv sync
uv run python -m src.preprocess.en_ibm_argq
```

## Synthetic data via vLLM

vLLM serves an OpenAI-compatible API, so the same client code targets either a
local or a remote server — only the URL changes.

```bash
# Local: launch vLLM on this machine
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Then in another shell:
cp .env.example .env  # edit if remote
export $(grep -v '^#' .env | xargs)
uv run python -m src.synth.counterargument  # smoke test
```

For a remote server, set `VLLM_BASE_URL=https://your-host/v1` in `.env` and
the same code runs unchanged. See `src/synth/counterargument.py` for the
sync entry point and `synthesize_counterarguments_async` for batched runs
with bounded concurrency.
