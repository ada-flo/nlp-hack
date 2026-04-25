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
