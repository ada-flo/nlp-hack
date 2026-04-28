## Selected candidates

| Candidate | Role | Korean support | English support | Recommended use |
|---|---|---:|---:|---|
| Self-pretrained OpenNMT-py LSTM seq2seq | Main baseline | Yes, if trained with Korean data | Yes | Best practical baseline |
| OpenNMT-py pretrained LSTM dialog model | Pretrained reference checkpoint | Unclear / not guaranteed | Likely | Sanity check and architecture reference |
| Korean-English LSTM/Seq2Seq repositories | Korean-focused implementation references | Yes | Yes | Reference for Korean tokenization, attention, and NMT-style pretraining |

## Candidate 1: Self-pretrained OpenNMT-py LSTM seq2seq

### Summary

This is the recommended main baseline. Instead of relying on a public pretrained checkpoint, train an LSTM encoder-decoder model on large auxiliary English and Korean dialogue or translation data, then fine-tune it on the debate-formatted dataset.

This option is the most defensible because:

- it satisfies the LSTM-based seq2seq requirement;
- it can support both English and Korean;
- it gives full control over tokenizer, vocabulary, special tokens, and data format;
- it avoids claiming the existence of a public bilingual pretrained LSTM checkpoint when such a checkpoint is difficult to verify.

### Characteristics

| Aspect | Recommendation |
|---|---|
| Framework | OpenNMT-py or equivalent PyTorch implementation |
| Encoder | 2-layer bidirectional LSTM |
| Decoder | 2-layer LSTM |
| Attention | Global attention recommended |
| Tokenization | SentencePiece unigram or BPE recommended |
| Vocabulary | Shared bilingual vocabulary if training one bilingual model |
| Special tokens | `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`, `<SEP>`, optionally `<EN>`, `<KO>` |
| Pretraining data | Larger dialogue, subtitle, or translation-style corpora |
| Fine-tuning data | Final debate JSONL dataset |

### When to use this baseline

Use this as the main baseline if the goal is to submit a model that is clearly compatible with the project rules and works for both English and Korean.

### Data format

The final debate data should be converted into source-target text files for OpenNMT-style training.

Input source:

```text
<LANG> topic <SEP> input_context
```

Target:

```text
target_output
```

Example:

```text
<KO> 안락사 허용 <SEP> 고통스러운 삶을 강제로 이어가게 하는 것은 비인도적입니다.
```

```text
하지만 생명의 가치는 타인이 판단할 수 없으며 남용의 소지가 큽니다.
```

### JSONL to source-target conversion

```python
# scripts/jsonl_to_opennmt.py

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_jsonl(input_path: Path, src_path: Path, tgt_path: Path, add_lang_tag: bool = True) -> None:
    src_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, \
         src_path.open("w", encoding="utf-8") as fsrc, \
         tgt_path.open("w", encoding="utf-8") as ftgt:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            lang = row.get("lang", "")
            topic = row["topic"].strip()
            input_context = row["input_context"].strip()
            target_output = row["target_output"].strip()

            prefix = f"<{lang.upper()}> " if add_lang_tag and lang else ""
            source = f"{prefix}{topic} <SEP> {input_context}"

            fsrc.write(source + "\n")
            ftgt.write(target_output + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--no-lang-tag", action="store_true")
    args = parser.parse_args()

    convert_jsonl(
        input_path=Path(args.input),
        src_path=Path(args.src),
        tgt_path=Path(args.tgt),
        add_lang_tag=not args.no_lang_tag,
    )
```

Usage:

```bash
python scripts/jsonl_to_opennmt.py \
  --input data/processed/train.jsonl \
  --src data/opennmt/train.src \
  --tgt data/opennmt/train.tgt

python scripts/jsonl_to_opennmt.py \
  --input data/processed/valid.jsonl \
  --src data/opennmt/valid.src \
  --tgt data/opennmt/valid.tgt

python scripts/jsonl_to_opennmt.py \
  --input data/processed/test.jsonl \
  --src data/opennmt/test.src \
  --tgt data/opennmt/test.tgt
```

### Suggested OpenNMT-py configuration

The exact OpenNMT-py command syntax can differ by version. Treat this as a starting configuration, not a guaranteed command for every installation.

```yaml
# configs/opennmt_lstm_baseline.yaml

save_data: data/opennmt/run/example
src_vocab: data/opennmt/vocab.src
share_vocab: true

src_seq_length: 160
tgt_seq_length: 120

data:
  corpus_1:
    path_src: data/opennmt/train.src
    path_tgt: data/opennmt/train.tgt
  valid:
    path_src: data/opennmt/valid.src
    path_tgt: data/opennmt/valid.tgt

save_model: checkpoints/lstm_debate/model

world_size: 1
gpu_ranks: [0]

encoder_type: brnn
decoder_type: rnn
rnn_type: LSTM
layers: 2
rnn_size: 512
word_vec_size: 300

global_attention: general
bridge: true

optim: adam
learning_rate: 0.001
batch_size: 64
valid_batch_size: 64
train_steps: 30000
valid_steps: 1000
save_checkpoint_steps: 1000

 dropout: 0.3
```

Correct the indentation before use. The `dropout` line should be aligned with the other top-level keys.

### Training procedure

1. Build or load the debate JSONL files.
2. Convert JSONL into source-target files.
3. Train or load a SentencePiece tokenizer if using subword tokenization.
4. Build vocabulary.
5. Pretrain on larger English/Korean auxiliary data.
6. Fine-tune on the debate dataset.
7. Evaluate separately on English and Korean test splits.
8. Report whether the checkpoint is randomly initialized, embedding-initialized, or self-pretrained.

### Strengths

- Most compliant with the LSTM seq2seq requirement.
- Supports English and Korean if trained on both.
- Easy to document in the final report.
- Fine-tuning and ablation experiments are straightforward.

### Weaknesses

- Not an off-the-shelf pretrained checkpoint.
- Requires more preprocessing and training time.
- Quality depends heavily on auxiliary pretraining data.
- LSTM models may still underperform Transformer models on long-context debate.

## Candidate 2: OpenNMT-py pretrained LSTM dialog model

### Summary

OpenNMT-py provides pretrained LSTM seq2seq examples, including a dialog model trained on OpenSubtitles-style data. This candidate is useful as a pretrained reference checkpoint or sanity-check baseline.

It should not be assumed to be Korean-capable unless the checkpoint vocabulary and training data confirm Korean coverage.

### Characteristics

| Aspect | Description |
|---|---|
| Framework | OpenNMT-py |
| Model type | LSTM encoder-decoder |
| Task | Dialogue generation |
| Likely training domain | Subtitle-style conversational text |
| Korean support | Not guaranteed |
| Best use | English-only sanity baseline, decoding reference, checkpoint-loading test |

### When to use this baseline

Use this candidate when developers need:

- a quick pretrained LSTM checkpoint to test inference;
- a reference for OpenNMT-py checkpoint loading;
- an English-oriented dialogue generation comparison;
- a sanity check before training the project-specific model.

Do not use it as the final bilingual baseline unless Korean vocabulary support is verified.

### Utilization procedure

1. Download the pretrained OpenNMT-py LSTM dialog checkpoint from the official OpenNMT model list.
2. Install the matching OpenNMT-py version if checkpoint compatibility issues occur.
3. Run inference on simple English dialogue inputs.
4. Test Korean inputs only as a diagnostic check.
5. If Korean outputs are broken or `<UNK>`-heavy, report the model as English-only.
6. Use the checkpoint as a reference, not as the primary Korean-English debate model.

### Example inference input format

If the pretrained model expects plain dialogue context, test both plain and project-formatted inputs:

```text
I think school uniforms should be mandatory because they reduce inequality.
```

```text
School uniforms <SEP> I think school uniforms should be mandatory because they reduce inequality.
```

For Korean diagnostic testing:

```text
교복 의무화 <SEP> 교복은 학생 간 경제적 차이를 줄일 수 있기 때문에 필요합니다.
```

If the model produces mostly unknown tokens or irrelevant output, do not treat it as Korean-capable.

### Strengths

- Real pretrained LSTM seq2seq checkpoint.
- Useful for validating the inference pipeline.
- Dialogue-oriented, unlike summarization or translation checkpoints.
- Good reference for OpenNMT-py model configuration.

### Weaknesses

- Korean support is uncertain.
- Domain is general dialogue, not debate.
- Vocabulary may not match the project tokenizer.
- Fine-tuning may be difficult if checkpoint and current OpenNMT-py versions are incompatible.

### Recommended role in the codebase

Use this as `baseline_reference_opennmt_dialog`, not as the main `lstm_debate_bilingual` model.

Suggested directory:

```text
checkpoints/
  opennmt_pretrained_dialog/
    README.md
    model.pt
    vocab.pt
```

Suggested metadata file:

```json
{
  "name": "opennmt_pretrained_lstm_dialog",
  "role": "reference_checkpoint",
  "architecture": "lstm_seq2seq",
  "task": "dialogue_generation",
  "korean_support_verified": false,
  "use_as_main_submission_model": false
}
```

## Candidate 3: Korean-English LSTM/Seq2Seq repositories

### Summary

Korean-English seq2seq repositories are useful as implementation references for Korean preprocessing, tokenization, attention-based encoder-decoder modeling, and translation-style pretraining.

These repositories should be treated cautiously as pretrained baselines. In many cases, the code is available but the trained checkpoint or original training data is not reliably downloadable.

### Representative repositories

| Repository type | Main value | Risk |
|---|---|---|
| Korean-English PyTorch seq2seq repository | Modern reference for Korean-English attention-based seq2seq training | Dataset may require separate approval or may not be redistributed |
| Korean-English MXNet/Gluon NMT repository | Older reference for Korean-English seq2seq implementation | Checkpoint availability may be unclear |

### Characteristics

| Aspect | Description |
|---|---|
| Model family | Seq2seq NMT, usually RNN/LSTM/GRU with attention |
| Language pair | Korean-English |
| Main use | Korean preprocessing and bilingual sequence modeling reference |
| Debate relevance | Indirect |
| Checkpoint reliability | Must be manually verified |
| Best role | Architecture and preprocessing reference |

### When to use this baseline

Use these repositories when developers need examples for:

- Korean tokenization before seq2seq training;
- Korean-English vocabulary construction;
- attention-based LSTM decoder implementation;
- teacher forcing training loops;
- inference with greedy or beam decoding;
- translation-style auxiliary pretraining.

Do not claim these are pretrained baselines unless the checkpoint is downloaded, loaded, and tested.

### Verification checklist

Before using one of these repositories as a checkpoint baseline, verify all of the following:

```text
[ ] The repository architecture is actually LSTM/GRU/RNN-based, not Transformer-based.
[ ] The pretrained checkpoint file is publicly available or locally reproducible.
[ ] The checkpoint loads without code modification.
[ ] The tokenizer or vocabulary files are available.
[ ] Korean input produces valid Korean or English output.
[ ] License and dataset terms allow the intended use.
[ ] The model can be adapted to the debate source-target format.
```

If any item fails, use the repository only as an implementation reference.

### How to adapt NMT-style code to debate generation

Most Korean-English seq2seq repositories expect translation pairs:

```text
source sentence -> target sentence
```

For the debate chatbot, replace this with:

```text
topic <SEP> opponent_argument -> rebuttal
```

Example Korean training pair:

```text
Source:
안락사 허용 <SEP> 고통스러운 삶을 강제로 이어가게 하는 것은 비인도적입니다.

Target:
하지만 생명의 가치는 타인이 판단할 수 없으며 남용의 소지가 큽니다.
```

Example English training pair:

```text
Source:
Legalization of Euthanasia <SEP> It is inhumane to force someone to continue a life full of suffering.

Target:
However, the value of life cannot be judged by others, and there is a high risk of misuse.
```

### Minimal PyTorch-style training loop pattern

This is a simplified structural reference. It is not a complete implementation.

```python
# Pseudocode only

for batch in train_loader:
    src_ids = batch["src_ids"]
    src_lengths = batch["src_lengths"]
    tgt_input_ids = batch["tgt_input_ids"]
    tgt_output_ids = batch["tgt_output_ids"]

    encoder_outputs, encoder_state = encoder(src_ids, src_lengths)
    logits = decoder(
        input_ids=tgt_input_ids,
        initial_state=encoder_state,
        encoder_outputs=encoder_outputs,
    )

    loss = sequence_cross_entropy(
        logits=logits,
        targets=tgt_output_ids,
        ignore_index=pad_id,
    )

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Korean preprocessing notes

For Korean, whitespace tokenization is often weak. Use one of the following:

| Option | Strength | Weakness |
|---|---|---|
| Character-level tokenization | Simple and robust | Longer sequences |
| SentencePiece | Good default for bilingual data | Requires training tokenizer |
| Morphological analyzer | Linguistically meaningful | Adds dependency and preprocessing complexity |
| Existing repository tokenizer | Matches reference implementation | May not generalize to debate data |

For this project, SentencePiece is the most practical default if time permits. Character-level tokenization is a reasonable fallback for a small hackathon model.

### Strengths

- Directly relevant to Korean-English sequence modeling.
- Useful for Korean preprocessing decisions.
- Provides implementation examples for attention-based seq2seq.
- Can inspire auxiliary translation pretraining.

### Weaknesses

- Often not debate-oriented.
- Public checkpoint availability is uncertain.
- Training data may be unavailable due to license or approval restrictions.
- Older repositories may require outdated dependencies.

### Recommended role in the codebase

Use these repositories as `reference_ko_en_seq2seq`, not as the default submitted model.

Suggested documentation location:

```text
docs/reference_models/ko_en_seq2seq_repos.md
```

Suggested metadata if a checkpoint is verified:

```json
{
  "name": "verified_ko_en_lstm_seq2seq_checkpoint",
  "role": "optional_checkpoint_baseline",
  "architecture": "lstm_or_gru_seq2seq",
  "task": "ko_en_translation",
  "checkpoint_verified": true,
  "tokenizer_verified": true,
  "use_as_main_submission_model": false
}
```
