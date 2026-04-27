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

## Training the LSTM seq2seq

Two encoder modes; both use a from-scratch LSTM decoder with Bahdanau attention.

```bash
# 1. Build the SentencePiece BPE vocab (32K, bilingual)
uv run python -m src.preprocess.build_vocab

# 2a. Phase 2 — BiLSTM encoder + LSTM decoder, ~30M params
uv run python -m src.train --encoder bilstm --epochs 10 --batch-size 32

# 2b. Phase 2 + FastText embeddings (download cc.en.300.vec, cc.ko.300.vec first)
uv run python -m src.train --encoder bilstm \
  --fasttext-en data/raw_manual/fasttext/cc.en.300.vec \
  --fasttext-ko data/raw_manual/fasttext/cc.ko.300.vec

# 3. Phase 3 — frozen XLM-R-base encoder + LSTM decoder, ~30M trainable
uv run python -m src.train --encoder xlmr --epochs 5 --batch-size 16
```

Each run writes to `checkpoints/<encoder>-<timestamp>/`:

- `args.json` — full hyperparameters
- `history.json` — per-epoch train_loss, valid_loss, perplexity, BLEU
- `best.pt` — best-BLEU checkpoint
- `test_metrics.json` — final test-set metrics

### Inference (beam search)

```bash
uv run python -m src.generate \
  --checkpoint checkpoints/bilstm-XXXX/best.pt \
  --topic "안락사 허용" \
  --input "고통스러운 삶을 강제로 이어가게 하는 것은 비인도적입니다." \
  --beam-size 4
```

## Encoder choice — read first

The brief says "LSTM-based seq2seq". The BiLSTM encoder is the strict
interpretation. The XLM-R encoder + LSTM decoder reads "may add or modify"
liberally — the decoder is still LSTM, but a transformer encoder may not
fly with a strict grader. **Verify with the TA before submitting xlmr.**

## Pushing the dataset to Hugging Face

Once `data/processed/` is populated:

```bash
# HF_TOKEN must have WRITE scope (https://huggingface.co/settings/tokens)
uv run python scripts/push_to_hf.py --repo-id <user>/<dataset-name>

# Private repo:
uv run python scripts/push_to_hf.py --repo-id <user>/<dataset-name> --private
```

The script uploads `train` / `validation` / `test` splits as parquet and
writes a dataset card (`README.md` in the HF repo) with schema, source
breakdown, and per-split language counts. See `scripts/push_to_hf.py`.

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
