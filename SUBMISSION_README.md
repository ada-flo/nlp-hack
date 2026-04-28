# Team 6 — Debate Battle LSTM Seq2Seq

Bilingual (Korean + English) debate-themed seq2seq chatbot. Encoder is a
**frozen XLM-RoBERTa-base**; decoder is a **2-layer LSTM with Bahdanau
attention**. Trained on ~46k pairs spanning IBM ArgQ, KLUE-NLI, casual
discourse corpora, and ~13k LLM-synthesized Korean debate pairs.

**Test BLEU: 3.13** on 5,443 held-out pairs (PPL 248).

Trained checkpoint + tokenizer:
[ada-flo/nlp-hack-debate-xlmr-lstm](https://huggingface.co/ada-flo/nlp-hack-debate-xlmr-lstm)

## Repo layout

```
src/
  train.py                    # training entrypoint (--encoder bilstm|xlmr)
  generate.py                 # beam-search inference
  data.py                     # bilingual dataloader + xlm-r variant
  model/
    lstm_seq2seq.py          # encoder/decoder/seq2seq
    embeddings.py            # FastText init helper (bilstm only)
  preprocess/                 # per-source adapters → data/interim/*.jsonl
    en_ibm_argq.py
    en_isotonic_conversation.py
    en_mc_conversation.py
    en_casual_conversation.py
    ko_korean_petitions.py
    ko_klue_nli.py
    ko_debate_synth.py        # motion-seeded debate-pair synth
    ko_aihub_seeded_synth.py  # AI-Hub topic-seeded debate synth (NEW)
    ko_aihub_topic_dialogue.py
    ko_aihub_purpose_dialog.py
    ko_aihub_persona_dialog.py
    merge_and_split.py        # dedupe + topic-level 80/10/10 split
    build_vocab.py            # train SentencePiece BPE
  synth/
    client.py, prompts.py, counterargument.py   # vLLM Qwen3-235B synth
scripts/
  run_pipeline.sh             # full data prep
  push_to_hf.py               # publish dataset
  push_model_to_hf.py         # publish model
  demo_inference.py           # quick CON-rebuttal demo
pyproject.toml, uv.lock
```

## Quick run

```bash
# Set up env (creates .venv via uv)
uv sync

# 1. Build training data into data/interim/*.jsonl, then merge
bash scripts/run_pipeline.sh

# 2. Train SentencePiece tokenizer (32k shared bilingual vocab)
uv run python -m src.preprocess.build_vocab

# 3. Train model (best results: xlmr encoder, warm-start chain)
uv run python -m src.train --encoder xlmr --epochs 5 --batch-size 16

# 4. Inference demo
uv run python scripts/demo_inference.py
```

## Training strategy

We chained **three** warm-start runs to land at our best checkpoint:

1. `xlmr-1777277398` — fresh xlmr training (5 ep, BLEU 1.70)
2. `xlmr-1777297357` — `--init-from xlmr-1777277398` (5 ep, BLEU 2.60)
3. `xlmr-1777348083` — `--init-from xlmr-1777297357` after expanding the
   training set with 6,206 fresh KO domain-seeded debate pairs from the
   AI-Hub topic taxonomy (+8k synth pairs total, both directions). 5 ep,
   **BLEU 3.13**.

Each warm-start improves over the previous best — the model picks up new
debate signal without forgetting prior knowledge.

## Data sources

| Source | Records (train) | Notes |
|---|---|---|
| IBM ArgQ 30K | 24,126 | Real EN PRO/CON pairs over ~70 motions |
| Korean Petitions (synth) | 8,203 | Petition body → Qwen3 counter-argument |
| ko_aihub_seeded_synth (NEW) | 6,206 | AI-Hub topic taxonomy → Qwen3 motion+PRO+CON |
| ko_debate_synth | 4,710 | 98 curated KO motions × 30 LLM PRO/CON |
| Isotonic / MC / Casual (EN) | 2,967 | Discourse corpora for fluency |

Synthesis prompts in `src/synth/prompts.py`. Synthesis used a vLLM-served
Qwen3-235B-A22B-Instruct-2507 over an SSH tunnel.

## Known limitations

- EN responses sometimes drift to canned argument templates (e.g. "we
  should not legalize the organ trade…") — most new training pairs are
  KO, so EN side benefits less from the recent data expansion.
- Generation is bounded at 64 tokens via beam search (beam=4).
- Multi-turn context (the optional bonus) is not implemented — each call
  conditions only on (topic, single-turn input).
