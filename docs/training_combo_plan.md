# Training Combo Plan

Sequential single-GPU (GPU 1) plan for the debate seq2seq hackathon.

Also see `docs/LSTM_Baseline_README.md` for the original candidate survey (Self-pretrained OpenNMT, OpenNMT pretrained dialog checkpoint, Korean-English LSTM repos). This plan operationalizes those candidates against the current repo.

## Constraints (from `Hackathon and Team Building.pdf`)

- LSTM-based seq2seq is mandatory as the **base** — encoder/architecture additions are allowed.
- Bilingual EN + KO.
- Optional bonus: multi-turn debate context.
- 3-hour competition window (this plan covers prep done before the window).

## On using an "already-pretrained" LSTM

`LSTM_Baseline_README.md` Candidate 2 (OpenNMT pretrained dialog) and Candidate 3 (KO-EN LSTM repos) both flag the same problem: **publicly available bilingual LSTM seq2seq checkpoints are unreliable**. Candidate 2 is English-only / Korean-uncertain; Candidate 3 checkpoints are often missing or non-redistributable. So the practical path to "pretrained LSTM" is:

1. **Self-pretrain (Candidate 1, recommended in the README).** Train our own LSTM on large auxiliary discourse corpora, then finetune on the 49K debate set. This is Step 4 below — the highest-ROI experiment in this plan.
2. **Pretrained representations into an LSTM decoder.** The `xlmr` encoder path (Step 3) is exactly this pattern: frozen pretrained encoder + LSTM decoder. The base is still LSTM (decoder side), satisfying the rule.
3. **Pretrained LSTM-LM encoder (ELMo / KoELMo).** Real pretrained *LSTM* (bidirectional LM) feature extractors exist. Investigation candidate; likely brittle infra and uncertain Korean coverage. Treat as a stretch goal only if Steps 3–4 are already strong.
4. **Pretrained word embeddings only (FastText cc.en/cc.ko).** Lightweight; already wired in `train.py` via `--fasttext-en/--fasttext-ko`. Smallest effort but smallest expected lift; deferred unless time permits.

Bottom line: for *this* dataset (bilingual debate), self-pretraining on the auxiliary corpora we already have is the most defensible "pretrained LSTM" approach. Step 4 is the concrete realization.

## Repo state today

- Encoders supported in `src/train.py`: `bilstm`, `xlmr` (XLM-R frozen → LSTM decoder).
- Data: 49K bilingual debate JSONL at `data/processed/{train,valid,test}.jsonl`.
- SentencePiece 32K shared vocab at `data/processed/spm.model`.
- Auxiliary discourse corpora are already preprocessed under `src/preprocess/` (not yet merged into a pretrain split).

## Run order

Each step is one job on GPU 1, run after the previous finishes.

### Step 1 — `bilstm` baseline (running)

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m src.train --encoder bilstm --epochs 10 --batch-size 32
```

- Purpose: baseline number to beat. From-scratch embeddings, no pretraining.
- ~190 s/epoch → ~32 min total.
- Record: best valid BLEU, test BLEU, run dir.

### Step 2 — sanity-check generations

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m src.generate \
  --checkpoint checkpoints/<bilstm-run>/best.pt \
  --topic "안락사 허용" \
  --input "고통스러운 삶을 강제로 이어가게 하는 것은 비인도적입니다."
```

- Purpose: eyeball output for fluency / topic-following / language matching, since BLEU alone hides degenerate output.

### Step 3 — `xlmr` encoder run

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m src.train --encoder xlmr --epochs 5 --batch-size 16
```

- Purpose: biggest single-shot upgrade with zero code change. Frozen XLM-R brings strong bilingual representations; LSTM decoder remains the generation head, satisfying the LSTM-base rule.
- Smaller batch size accounts for XLM-R memory cost.
- Expected: noticeably higher BLEU than Step 1, especially on Korean validation.

### Step 4 — curriculum pretrain → finetune (highest-ROI; needs prep work)

Sequence:

1. Pretrain `bilstm` on merged auxiliary discourse corpora.
2. Continue-train (finetune) on the 49K debate split.

Required prep before this step can run:

- Add `--init-from CHECKPOINT` flag to `src/train.py` (~10 lines: load weights from a prior `best.pt`, skip optimizer state).
- Build `data/pretrain/{train,valid}.jsonl` by merging into the same schema:
  - `en_isotonic_conversation`
  - `en_casual_conversation`
  - `ko_aihub_topic_dialogue`
  - `en_ibm_argq` (argument quality — closest discourse signal to debate)
- Reuse the existing `spm.model` (do not retrain the tokenizer; keeps vocab consistent across phases).

Run:

```bash
# Phase A: pretrain on auxiliary corpora
CUDA_VISIBLE_DEVICES=1 uv run python -m src.train \
  --encoder bilstm --epochs 3 --batch-size 32 \
  --data-dir data/pretrain --run-name pretrain-bilstm

# Phase B: finetune on debate data
CUDA_VISIBLE_DEVICES=1 uv run python -m src.train \
  --encoder bilstm --epochs 10 --batch-size 32 \
  --init-from checkpoints/pretrain-bilstm/best.pt \
  --run-name finetune-bilstm
```

- Purpose: the PDF explicitly flags 10K is too small. Auxiliary discourse corpora teach basic conversational structure before the model sees the narrow debate distribution.
- Expected: should beat both Step 1 and Step 3 if the auxiliary mix is reasonable.

### Step 5 (optional) — multi-turn context encoder

Only if Steps 1–4 finished and there is time left.

- Reformat encoder input to `<turn>prev_user</turn> <turn>prev_bot</turn> ... <SEP> current_input` using debate history.
- Most teams will only do single-turn; this is the differentiator the PDF hints at.
- Requires data-side change (history-aware JSONL builder), not model-side.

### Step 6 (optional) — persona/stance token conditioning

Cheap controllability win (~30 min):

- Prepend `<PRO>` / `<CON>` / `<KO>` / `<EN>` tokens to encoder input.
- Matches the "critical persona" framing in the deck title.
- Add tokens to SP user-defined-symbols list and rebuild only if a clean slate is desired; otherwise reserve unused vocab slots and inject as plain BPE pieces.

### Step 7 (stretch, only if Steps 1–4 are done) — pretrained LSTM-LM encoder (ELMo / KoELMo)

Investigation, not a guaranteed run:

- ELMo and Korean variants (KoELMo) are real pretrained *LSTM* bidirectional LMs. Could replace `BiLSTMEncoder` with a frozen ELMo encoder feeding the existing LSTM decoder.
- Risks: package compatibility (older AllenNLP-era code), Korean checkpoint availability, vocab/tokenizer alignment with our SentencePiece model.
- Decide go/no-go after a 30-minute feasibility check: does a working ELMo or KoELMo checkpoint load at all in this venv?

## Skip / lower priority

- **OpenNMT pretrained dialog checkpoint** (Candidate 2 in `LSTM_Baseline_README.md`): Korean support not guaranteed; vocab/tokenizer mismatch with our SP model; infra friction. Only useful as an English sanity check, not as our submission.
- **Korean-English LSTM/seq2seq repo checkpoints** (Candidate 3): public checkpoint availability is unreliable per the README's verification checklist. Reference for code patterns only.
- **FastText init**: ~2 GB per language to download; expected gain is small relative to Steps 3–4.
- **Deeper LSTM stacks (4+ layers)**: bigger compute cost, marginal at this dataset size.
- **Copy/coverage attention**: real implementation cost, not a clear hackathon win.
- **Knowledge distillation from a teacher LLM**: large data-prep lift; do only if Step 4 is already strong and there is time to spare.

## Numbers to record

For each run, capture in a results table:

- Run dir + encoder
- Epochs, batch size
- Best valid BLEU (and epoch)
- Test BLEU (overall + per-language EN/KO if reported)
- Sample generations: 2 KO, 2 EN, including at least one short input

This table goes straight into the technical report.
