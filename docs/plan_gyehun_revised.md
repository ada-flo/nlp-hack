# Debate Chatbot Dataset Pipeline

This repository prepares English and Korean training data for a debate-themed LSTM-based seq2seq chatbot.

The main goal of this README is to define:

- starting-point datasets;
- how each dataset can be converted into debate-style training examples;
- a common preprocessing pipeline;
- a codebase structure for later implementation.

Dataset proportions are intentionally **not fixed** here. Final source ratios, language balance, and total sample counts should be controlled through configuration, not hard-coded in preprocessing scripts.

## Project constraints

The team project requires a debate-themed dialogue generation model using an LSTM-based seq2seq base model. The training examples should follow this structure:

```json
{
  "topic": "Legalization of Euthanasia",
  "input_context": "It is inhumane to force someone to continue a life full of suffering.",
  "target_output": "However, the value of life cannot be judged by others, and there is a high risk of misuse.",
  "encoder_input": "Legalization of Euthanasia <SEP> It is inhumane to force someone to continue a life full of suffering.",
  "decoder_input": "<SOS> However, the value of life cannot be judged by others, and there is a high risk of misuse.",
  "decoder_target": "However, the value of life cannot be judged by others, and there is a high risk of misuse. <EOS>"
}
```

Data may be directly collected, transformed, or synthesized, but all synthetic records should retain source and prompt metadata.

## Final JSONL schema

Final processed files should be stored as JSONL:

```text
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/test.jsonl
```

Each line should contain one record:

```json
{
  "id": "en_ibm_argq_000001",
  "lang": "en",
  "source": "ibm_argq_30k",
  "topic": "We should ban cosmetic surgery for minors",
  "input_context": "A minor does not have the maturity to decide on procedures with long-term consequences.",
  "target_output": "That concern is valid, but a strict ban may ignore cases where medical or psychological needs justify supervised treatment.",
  "encoder_input": "We should ban cosmetic surgery for minors <SEP> A minor does not have the maturity to decide on procedures with long-term consequences.",
  "decoder_input": "<SOS> That concern is valid, but a strict ban may ignore cases where medical or psychological needs justify supervised treatment.",
  "decoder_target": "That concern is valid, but a strict ban may ignore cases where medical or psychological needs justify supervised treatment. <EOS>",
  "meta": {
    "input_stance": "pro",
    "target_stance": "con",
    "is_synthetic": true,
    "source_record_ids": ["example_source_id"],
    "synthesis_prompt_version": "v1"
  }
}
```

Required model fields:

- `topic`
- `input_context`
- `target_output`
- `encoder_input`
- `decoder_input`
- `decoder_target`

Recommended metadata fields:

- `id`
- `lang`
- `source`
- `input_stance`
- `target_stance`
- `is_synthetic`
- `source_record_ids`
- `synthesis_prompt_version`

## Starting point datasets

### English datasets

| Dataset | Link | Dataset type | Debate-data utilization |
|---|---|---|---|
| Winning Arguments / ChangeMyView Corpus | [ConvoKit documentation](https://convokit.cornell.edu/documentation/winning.html) | Real persuasive Reddit discussion threads | Use parent-child reply pairs. Use the thread title or opening post as `topic`, the parent comment as `input_context`, and the reply as `target_output`. Prefer replies with disagreement, persuasion, or rebuttal signals. |
| IBM Argument Quality Ranking 30K | [Hugging Face dataset](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k) | Topic, argument, quality score, stance label | Pair arguments with the same topic and opposite `stance_WA`. Use high-quality arguments first. Convert `pro -> con` and optionally `con -> pro`. |
| args.me corpus | [Hugging Face dataset](https://huggingface.co/datasets/webis/args_me), [Webis page](https://webis.de/data/args-me-corpus.html) | Debate portal arguments | Use the claim or conclusion as `topic`. Pair pro and con arguments under the same claim. Useful for broad topic coverage. |
| PERSPECTRUM | [ACL Anthology](https://aclanthology.org/N19-1053/) | Claims, perspectives, evidence | Use the claim as `topic`. Use one perspective as `input_context` and an opposing perspective as `target_output`. Evidence can support LLM-generated rebuttals. |
| DebateSum | [ACL Anthology](https://aclanthology.org/2020.argmining-1.1/), [GitHub repository](https://github.com/Hellisotherpeople/DebateSum), [Hugging Face dataset](https://huggingface.co/datasets/Hellisotherpeople/DebateSum) | Debate evidence and summaries | Convert evidence-summary pairs into concise argument/rebuttal samples. Long evidence should be summarized before seq2seq training. |
| Argument Annotated Essays v2 / AAE2 | [Hugging Face wrapper](https://huggingface.co/datasets/pie/aae2), [TU Darmstadt dataset page](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) | Persuasive essays with argument components and relations | Extract claim-premise and support/attack relations. Convert attack relations into rebuttal-style examples. |
| Schopenhauer Debate | [Hugging Face dataset](https://huggingface.co/datasets/raphaaal/schopenhauer-debate) | Small synthetic debate examples | Use mainly as a formatting/style reference. Do not rely on it as a main training source because it is small and synthetic. |

### Korean datasets

| Dataset | Link | Dataset type | Debate-data utilization |
|---|---|---|---|
| K-News-Stance | [GitHub repository](https://github.com/ssu-humane/K-News-Stance) | Korean news stance data | Use `issue` as `topic`. Use headline, lead, quotation, conclusion, or article segment as `input_context`. Pair supportive and oppositional segments where possible. Otherwise synthesize the missing rebuttal. The repository states that the dataset must be requested through a download form. |
| Korean Petitions / 청와대 국민청원 | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/korean_petitions.html) | Korean social and policy petitions | Use petition `title` as a debate motion. Use petition `text` as a source argument. Generate a concise opposing response. Filter personal information and low-quality petitions. |
| AI Hub Korean Dialogue Summarization / 한국어 대화 요약 | [AI Hub dataset](https://aihub.or.kr/aidata/30714) | Korean dialogue originals and one-sentence summaries | Filter debate-like, discussion-like, or social-topic dialogues. Use adjacent argumentative turns directly, or use the summary/topic to guide LLM synthesis. |
| AI Hub Topic-wise Everyday Text Dialogue / 주제별 텍스트 일상 대화 데이터 | [AI Hub dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=543&topMenu=100) | Korean everyday dialogue with topic and speech-act labels | Use for conversational fluency and everyday debate topics. Filter opinion, disagreement, suggestion, and explanation turns. |
| AI Hub Meeting Speech / 회의 음성 | [AI Hub dataset](https://aihub.or.kr/aidata/30709) | Korean meeting, current-affairs, discussion, and talk transcripts | Use transcript text, not audio. Prioritize current-affairs, talk, and debate-like segments. Remove moderator-only turns, greetings, and filler-heavy utterances. |
| AI Hub Korean Dialogue / 한국어 대화 | [AI Hub dataset](https://aihub.or.kr/aidata/85) | Korean public-service and small-business dialogue | Use only as auxiliary dialogue data. It can improve Korean turn-taking but is not primarily debate-oriented. |
| AI Hub Korean Dialogue Dataset / 한국어 대화 데이터셋 | [AI Hub dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=272&topMenu=100) | Korean emergency and office multi-turn dialogue | Use only as auxiliary Korean dialogue data. It should not dominate the final debate dataset. |

### Additional Korean datasets available without an AI Hub key

These sources are useful when AI Hub access is delayed or unavailable. They are not all debate datasets. Treat them as **topic sources**, **opinion sources**, **dialogue-style sources**, or **safety filters**, and keep the conversion method explicit in metadata.

| Dataset | Link | Access note | Dataset type | Debate-data utilization |
|---|---|---|---|---|
| Korean Chatbot Data / 챗봇 문답 페어 | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/korean_chatbot_data.html), [GitHub repository](https://github.com/songys/Chatbot_data) | Directly downloadable through Korpora | Korean QA-style chat pairs with daily-life/farewell/love labels | Use only as auxiliary Korean turn-style data. It is not argumentative. Keep direct `question -> answer` pairs separate from debate pairs, or rewrite the question as a simple opinion prompt and synthesize an opposing answer. |
| KcBERT comments / KcBERT 댓글 데이터 | [Korpora documentation](https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/korean_comments.html), [GitHub repository](https://github.com/Beomi/KcBERT) | Directly downloadable through Korpora; large corpus | Korean online news comments | Use as a large pool of informal Korean opinions. Select comments containing stance markers, disagreement, claims, or policy nouns; then generate a concise counterargument. Apply strict toxicity and privacy filtering. |
| KorNLI | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/kornli.html), [GitHub repository](https://github.com/kakaobrain/KorNLUDatasets) | Directly downloadable through Korpora or GitHub | Korean natural language inference pairs | Use `contradiction` pairs as weak rebuttal seeds: `premise -> hypothesis` or `hypothesis -> premise`. Add a templated or LLM-generated connective phrase only when needed. Use `entailment` pairs for paraphrase and topic-normalization support, not rebuttals. |
| KLUE-NLI and KLUE-YNAT | [Hugging Face dataset](https://huggingface.co/datasets/klue/klue), [GitHub repository](https://github.com/KLUE-benchmark/KLUE) | Hugging Face / GitHub | Korean NLU benchmark: NLI, topic classification, and other tasks | Use KLUE-NLI contradiction examples for compact Korean rebuttal pairs. Use YNAT news headlines and topic labels as Korean debate-topic seeds. Do not treat YNAT as dialogue data. |
| NAVER Sentiment Movie Corpus / NSMC | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/nsmc.html), [Hugging Face dataset](https://huggingface.co/datasets/e9t/nsmc), [GitHub repository](https://github.com/e9t/nsmc) | Directly downloadable through Korpora or Hugging Face | Korean movie-review sentiment data | Use as everyday-opinion material. Since it lacks explicit debate structure, use review text as `input_context` and synthesize a short opposing reply. Keep movie-review debate separate from public-policy debate. |
| Korean Question Pair / 한국어 질문쌍 | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/question_pair.html), [GitHub repository](https://github.com/songys/Question_pair) | Directly downloadable through Korpora | Korean paired questions with same/different-question labels | Use for topic paraphrase, duplicate detection, and motion normalization. It is not a rebuttal dataset; avoid using the pair itself as `target_output` unless the goal is paraphrase pretraining. |
| KOTE: Korean Online That-gul Emotions | [GitHub repository](https://github.com/searle-j/KOTE), [Hugging Face dataset](https://huggingface.co/datasets/searle-j/kote), [ACL Anthology paper](https://aclanthology.org/2024.lrec-main.1499/) | GitHub / Hugging Face | Korean online comments labeled with fine-grained emotions | Select comments labeled with complaint, doubt, anger, distrust, disappointment, or interest as opinion inputs. Generate a calm opposing response. Use emotion labels to avoid overrepresenting hostile or highly emotional examples. |
| Korean Hate Speech Dataset | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/korean_hate_speech.html), [GitHub repository](https://github.com/kocohub/korean-hate-speech) | Directly downloadable through Korpora or GitHub | Korean toxic-speech/news-comment corpus with labeled and unlabeled subsets | Use primarily as a safety filter and negative-control corpus. Do not train the debate model to imitate hateful or offensive text. Optionally use clean/non-hate examples as informal Korean comment style after filtering. |
| K-MHaS | [GitHub repository](https://github.com/adlnlp/K-MHaS), [Hugging Face dataset](https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech), [ACL Anthology paper](https://aclanthology.org/2022.coling-1.311/) | GitHub / Hugging Face | Korean multi-label hate-speech news-comment corpus | Use as an additional safety filter for politics, gender, age, religion, race, origin, physical appearance, and profanity categories. It is more useful for filtering generated debate data than for positive training examples. |
| KOLD: Korean Offensive Language Dataset | [GitHub repository](https://github.com/boychaboy/KOLD), [Hugging Face dataset](https://huggingface.co/datasets/nayohan/KOLD), [ACL Anthology paper](https://aclanthology.org/2022.emnlp-main.744/) | GitHub / Hugging Face | Korean offensive-language comments with contextual titles | Use article/video titles as weak topics and comments as inputs only after filtering. Prefer using it to reject toxic generated outputs and to test whether the chatbot avoids offensive replies. |
| KoPolitic Benchmark Dataset | [GitHub repository](https://github.com/Kdavid2355/KoPolitic-Benchmark-Dataset), [arXiv paper](https://arxiv.org/abs/2311.01712) | GitHub / Google Drive link in repository | Korean political news articles with political-orientation and pro-government labels | Summarize an article into a claim, use the orientation/pro-government labels as stance metadata, and synthesize an opposing response. Useful for Korean political topics, but outputs must be checked for factuality and neutrality. |
| KorNAT Social Values | [Hugging Face dataset](https://huggingface.co/datasets/jiyounglee0523/KorNAT), [ACL Anthology paper](https://aclanthology.org/2024.findings-acl.666/) | Hugging Face | Korean social-value and common-knowledge multiple-choice benchmark | Use social-value questions as debate motions. Convert answer choices into stance descriptions such as `strongly agree`, `agree`, `neutral`, `disagree`, and `strongly disagree`; then synthesize short arguments for opposing choices. Do not use common-knowledge questions as debate data unless they are converted into value or policy questions. |
| Team Popong Data for Political R&D | [GitHub repository](https://github.com/teampopong/data-for-rnd) | GitHub | Korean political data: bills, candidates, pledges, parties, cosponsorship | Use bill names and campaign pledges as Korean public-policy topic seeds. Generate balanced pro/con examples. This is a topic source, not a natural-language rebuttal corpus. |
| NIKL Modu: Newspaper / 모두의 말뭉치 신문 | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/modu_news.html), [National Institute of Korean Language corpus portal](https://corpus.korean.go.kr/) | No AI Hub key, but requires NIKL/manual corpus access; Korpora provides loading only | Korean newspaper paragraphs with metadata | Use article topic and paragraph text as evidence for generating factual Korean debate claims. Summarize long paragraphs before seq2seq training. |
| NIKL Modu: Messenger / 모두의 말뭉치 메신저 | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/modu_messenger.html), [National Institute of Korean Language corpus portal](https://corpus.korean.go.kr/) | No AI Hub key, but requires NIKL/manual corpus access; Korpora provides loading only | Korean messenger-style multi-turn conversations | Use adjacent turns for Korean conversational style. Only convert to debate examples if the turns contain opinion, disagreement, justification, or correction. |
| OpenSubtitles2016 Korean-English | [Korpora documentation](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/open_subtitles.html), [OPUS/OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles.php) | Directly downloadable through Korpora | Korean-English subtitle pairs | Use only as auxiliary conversational Korean or bilingual alignment data. It is noisy, fictional, and not debate-specific, so keep it separate from final debate evaluation. |

Recommended priority without AI Hub access:

```text
1. Korean Petitions + K-News-Stance if available by form
2. KorNLI / KLUE-NLI contradiction pairs
3. KcBERT comments + KOTE comments with safety filtering
4. KoPolitic / KorNAT / Team Popong topic seeds
5. Korean Chatbot Data, Modu Messenger, or OpenSubtitles as auxiliary dialogue-style data
6. Hate/offensive-language corpora as filters, not positive debate targets
```

## Dataset utilization strategies

### 1. Direct dialogue extraction

Use for ChangeMyView, AI Hub dialogue datasets, meeting transcripts, and any multi-turn corpus.

```text
topic          = thread title, issue label, summary, or normalized topic
input_context  = previous turn / opponent turn
target_output  = next turn / response turn
```

Reject pairs when:

- either side is empty;
- either side is only a greeting, agreement marker, or moderation text;
- the reply does not address the input;
- the text contains private information, usernames, or excessive profanity;
- the pair is too long for an LSTM seq2seq model.

### 2. Stance-pair construction

Use for IBM ArgQ 30K, args.me, K-News-Stance, and PERSPECTRUM.

```text
topic/claim/issue -> topic
pro argument      -> input_context
con argument      -> target_output
```

The direction can also be reversed:

```text
con argument      -> input_context
pro argument      -> target_output
```

Keep the direction explicit in metadata:

```json
{
  "meta": {
    "input_stance": "pro",
    "target_stance": "con"
  }
}
```

### 3. LLM-assisted synthesis

Use when a source provides a topic and one argument but no direct rebuttal. Korean Petitions and some news data will likely require this.

Recommended synthesis prompt:

```text
You are generating training data for a debate chatbot.
Language: {language}
Topic: {topic}
Opponent argument: {input_context}

Write one concise opposing response in the same language.
Requirements:
- Directly address the opponent argument.
- Use one clear counter-reason.
- Keep the output to one or two sentences.
- Do not insult the speaker.
- Do not invent statistics, names, dates, legal claims, or factual claims not grounded in the given text.
Return only the response text.
```

Synthetic examples must include:

```json
{
  "meta": {
    "is_synthetic": true,
    "synthesis_prompt_version": "v1",
    "source_record_ids": ["..."]
  }
}
```

### Non-AI-Hub Korean source conversions

Use the following conversions for the additional Korean datasets listed above.

**NLI contradiction conversion**

Use for KorNLI and KLUE-NLI. The cleanest direct conversion is:

```text
topic          = "두 문장의 관점 차이" or a short LLM-generated topic
input_context  = premise
target_output  = hypothesis, only when label == contradiction
```

A more debate-like conversion is to keep the premise as the opponent claim and generate a one-sentence rebuttal grounded in the contradiction pair:

```text
topic          = extracted topic from premise + hypothesis
input_context  = premise
target_output  = "그 주장에는 한계가 있습니다. ..."
```

Do not use `entailment` pairs as rebuttals. Use them for paraphrase normalization or duplicate detection.

**Opinion/comment-to-rebuttal conversion**

Use for KcBERT comments, KOTE, NSMC, Korean Hate Speech clean examples, and KOLD/K-MHaS clean or non-offensive examples.

```text
topic          = article title, emotion label, review domain, or generated short issue label
input_context  = selected opinion/comment/review
target_output  = synthetic opposing reply
```

Filtering is mandatory. Reject examples that include personal attacks, slurs, direct insults, private identifiers, or ungrounded factual accusations.

**Topic-seed conversion**

Use for KoPolitic, KorNAT, Team Popong bills/pledges, NIKL newspaper, and KLUE-YNAT.

```text
topic          = bill name, pledge, news headline, social-value question, or normalized policy motion
input_context  = short argument generated for one stance
target_output  = short opposing argument generated for the opposite stance
```

Store the original source string in `meta.source_record_ids` or a separate metadata field so that synthetic examples remain auditable.

**Safety-filter conversion**

Use Korean Hate Speech Dataset, K-MHaS, and KOLD as filters before training:

```text
candidate_pair -> toxicity/offensiveness filter -> accepted_pair or rejected_pair
```

These corpora should not become positive target-output data unless the selected row is explicitly clean/non-offensive and manually inspected.

### 4. Translation-based augmentation

Translation can supplement the dataset, but it should not replace Korean-native data.

Acceptable uses:

- translate a small subset of English debate pairs into Korean for style balancing;
- translate Korean topics into English for bilingual topic normalization;
- mark every translated example in metadata.

Recommended metadata:

```json
{
  "meta": {
    "is_translated": true,
    "translation_direction": "en_to_ko"
  }
}
```

## Preprocessing pipeline

### Step 1. Ingest raw data

Do not commit raw datasets to Git. Store them locally:

```text
data/raw/en/
data/raw/ko/
data/raw_manual/aihub/
```

Recommended `.gitignore` entries:

```text
data/raw/
data/raw_manual/
data/interim/
*.zip
*.tar.gz
*.parquet
*.csv
```

### Step 2. Convert to canonical intermediate records

Before final seq2seq formatting, convert all sources into a common intermediate schema:

```json
{
  "source": "k_news_stance",
  "lang": "ko",
  "topic": "개 식용 금지법 국회 본회의 통과",
  "text": "...",
  "stance": "supportive",
  "source_record_id": "..."
}
```

### Step 3. Normalize text

Apply:

- Unicode normalization with `NFKC`;
- whitespace normalization;
- URL and email removal;
- username and handle removal;
- HTML entity decoding;
- removal of duplicated punctuation where excessive.

For Korean:

- preserve Hangul and Korean punctuation;
- remove obvious OCR/STT noise from transcripts;
- remove personal names or sensitive identifiers when present.

### Step 4. Build debate pairs

Create pairs using one of these methods:

- adjacent dialogue turns;
- opposite-stance argument matching;
- LLM-generated rebuttals.

Each pair must have:

```text
topic
input_context
target_output
```

### Step 5. Filter and deduplicate

Suggested configurable filters:

```yaml
filters:
  min_input_chars: 20
  min_target_chars: 20
  max_input_chars: 600
  max_target_chars: 600
  max_encoder_tokens: 120
  max_decoder_tokens: 120
  reject_same_input_target: true
  deduplicate_by_normalized_text: true
```

These values are starting defaults, not hard requirements. LSTM seq2seq models usually benefit from shorter sequences, so long evidence documents should be summarized before training.

### Step 6. Create seq2seq fields

Use the slide-compatible format:

```text
encoder_input  = topic + " <SEP> " + input_context
decoder_input  = "<SOS> " + target_output
decoder_target = target_output + " <EOS>"
```

If training one bilingual model, optionally prefix the encoder with a language tag:

```text
encoder_input = "<KO> " + topic + " <SEP> " + input_context
encoder_input = "<EN> " + topic + " <SEP> " + input_context
```

Do not add the language tag unless the tokenizer and model code are prepared to handle it.

### Step 7. Split by topic

Use topic-level splitting, not random row-level splitting. Random splitting can leak the same topic or near-duplicate arguments across train, validation, and test.

```text
unique topics -> shuffle with fixed seed -> assign topics to train/valid/test
```

## Minimal preprocessing code

The following code is intentionally simple. It is suitable for an initial codebase and can later be split into source-specific adapters.

```python
# src/preprocess/common.py

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SEP = "<SEP>"
SOS = "<SOS>"
EOS = "<EOS>"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text while preserving English and Korean content."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def stable_id(*parts: str, prefix: str = "record") -> str:
    raw = "||".join(parts).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_seq2seq_record(
    *,
    lang: str,
    source: str,
    topic: str,
    input_context: str,
    target_output: str,
    meta: Optional[Dict[str, Any]] = None,
    add_language_tag: bool = False,
) -> Optional[Dict[str, Any]]:
    topic = normalize_text(topic)
    input_context = normalize_text(input_context)
    target_output = normalize_text(target_output)

    if not topic or not input_context or not target_output:
        return None
    if input_context == target_output:
        return None

    lang_tag = f"<{lang.upper()}> " if add_language_tag else ""
    encoder_input = f"{lang_tag}{topic} {SEP} {input_context}"
    decoder_input = f"{SOS} {target_output}"
    decoder_target = f"{target_output} {EOS}"

    return {
        "id": stable_id(lang, source, topic, input_context, target_output, prefix=source),
        "lang": lang,
        "source": source,
        "topic": topic,
        "input_context": input_context,
        "target_output": target_output,
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "meta": meta or {},
    }


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

Example adapter for IBM ArgQ 30K:

```python
# src/preprocess/en_ibm_argq.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from common import make_seq2seq_record, write_jsonl


def stance_name(value: int) -> str:
    if value == 1:
        return "pro"
    if value == -1:
        return "con"
    return "neutral"


def build_records() -> List[dict]:
    dataset = load_dataset(
        "ibm-research/argument_quality_ranking_30k",
        "argument_quality_ranking",
    )

    rows = []
    for split_name, split in dataset.items():
        for row in split:
            rows.append({**row, "split": split_name})

    grouped: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stance = int(row["stance_WA"])
        if stance not in {1, -1}:
            continue
        grouped[row["topic"]][stance].append(row)

    records = []
    for topic, by_stance in tqdm(grouped.items()):
        pros = sorted(by_stance[1], key=lambda x: float(x.get("WA", 0.0)), reverse=True)
        cons = sorted(by_stance[-1], key=lambda x: float(x.get("WA", 0.0)), reverse=True)

        # Pair opposite stances without fixing the final sample count here.
        for pro, con in zip(pros, cons):
            for input_row, target_row in [(pro, con), (con, pro)]:
                record = make_seq2seq_record(
                    lang="en",
                    source="ibm_argq_30k",
                    topic=topic,
                    input_context=input_row["argument"],
                    target_output=target_row["argument"],
                    meta={
                        "input_stance": stance_name(int(input_row["stance_WA"])),
                        "target_stance": stance_name(int(target_row["stance_WA"])),
                        "is_synthetic": False,
                        "source_record_ids": [],
                        "quality_input_WA": float(input_row.get("WA", 0.0)),
                        "quality_target_WA": float(target_row.get("WA", 0.0)),
                    },
                )
                if record:
                    records.append(record)

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/en_ibm_argq.jsonl")
```

Example adapter skeleton for Korean Petitions:

```python
# src/preprocess/ko_korean_petitions.py

from __future__ import annotations

from typing import List

from Korpora import Korpora

from common import make_seq2seq_record, write_jsonl


def summarize_petition_as_argument(text: str, max_chars: int = 280) -> str:
    """Temporary heuristic. Replace with a summarizer or LLM step later."""
    return text.strip()[:max_chars]


def synthesize_counterargument(topic: str, input_context: str) -> str | None:
    """
    Placeholder for an LLM call.
    Replace this function with the selected LLM client.
    Return None until the generation step is implemented.
    """
    return None


def build_records() -> List[dict]:
    corpus = Korpora.load("korean_petitions")
    records = []

    for item in corpus.train:
        topic = item.title or item.category
        input_context = summarize_petition_as_argument(item.text)
        target_output = synthesize_counterargument(topic, input_context)
        if target_output is None:
            continue

        record = make_seq2seq_record(
            lang="ko",
            source="korean_petitions",
            topic=topic,
            input_context=input_context,
            target_output=target_output,
            meta={
                "input_stance": "petition_position",
                "target_stance": "opposition",
                "is_synthetic": True,
                "synthesis_prompt_version": "v1",
                "category": item.category,
                "num_agree": item.num_agree,
                "begin": item.begin,
                "end": item.end,
            },
        )
        if record:
            records.append(record)

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/ko_korean_petitions.jsonl")
```

## Additional non-AI-Hub Korean preprocessing examples

Several of the selected no-key Korean sources can be fetched through Korpora:

```bash
korpora fetch --corpus korean_petitions
korpora fetch --corpus korean_chatbot_data
korpora fetch --corpus kcbert
korpora fetch --corpus kornli
korpora fetch --corpus nsmc
korpora fetch --corpus question_pair
korpora fetch --corpus korean_hate_speech
korpora fetch --corpus open_substitles
```

The `open_substitles` spelling follows the current Korpora corpus key shown in the Korpora documentation.

Example adapter skeleton for KorNLI contradiction pairs:

```python
# src/preprocess/ko_kornli.py

from __future__ import annotations

from typing import Iterable, List

from Korpora import Korpora

from common import make_seq2seq_record, write_jsonl


def iter_available_splits(corpus: object) -> Iterable[tuple[str, Iterable[object]]]:
    """Korpora split names differ by corpus; this keeps the adapter defensive."""
    candidate_names = [
        "multinli_train",
        "snli_train",
        "xnli_dev",
        "xnli_test",
        "train",
        "dev",
        "test",
    ]
    for name in candidate_names:
        split = getattr(corpus, name, None)
        if split is not None:
            yield name, split


def is_contradiction(label: object) -> bool:
    return str(label).strip().lower() in {"contradiction", "contradictory", "2"}


def build_records() -> List[dict]:
    corpus = Korpora.load("kornli")
    records = []

    for split_name, split in iter_available_splits(corpus):
        for item in split:
            if not is_contradiction(getattr(item, "label", "")):
                continue

            premise = getattr(item, "text", "")
            hypothesis = getattr(item, "pair", "")

            record = make_seq2seq_record(
                lang="ko",
                source="kornli",
                topic="문장 관계 반박",
                input_context=premise,
                target_output=hypothesis,
                meta={
                    "conversion": "nli_contradiction_pair",
                    "input_stance": "premise",
                    "target_stance": "contradiction",
                    "is_synthetic": False,
                    "split": split_name,
                },
            )
            if record:
                records.append(record)

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/ko_kornli.jsonl")
```

Example adapter skeleton for KcBERT comments as opinion inputs:

```python
# src/preprocess/ko_kcbert_comments.py

from __future__ import annotations

from typing import Iterable, List

from Korpora import Korpora

from common import make_seq2seq_record, normalize_text, write_jsonl

STANCE_MARKERS = (
    "찬성", "반대", "문제", "필요", "안된다", "해야", "하지마", "왜", "정책", "세금", "교육", "정부"
)


def looks_opinionated(text: str) -> bool:
    text = normalize_text(text)
    return len(text) >= 20 and any(marker in text for marker in STANCE_MARKERS)


def synthesize_counterargument(topic: str, input_context: str) -> str | None:
    """Replace with the selected LLM client. Return None until generation is implemented."""
    return None


def build_records(max_candidates: int | None = None) -> List[dict]:
    corpus = Korpora.load("kcbert")
    records = []
    seen = 0

    for comment in corpus.train:
        comment = normalize_text(str(comment))
        if not looks_opinionated(comment):
            continue

        seen += 1
        if max_candidates is not None and seen > max_candidates:
            break

        topic = "온라인 뉴스 댓글 쟁점"
        target_output = synthesize_counterargument(topic, comment)
        if target_output is None:
            continue

        record = make_seq2seq_record(
            lang="ko",
            source="kcbert_comments",
            topic=topic,
            input_context=comment,
            target_output=target_output,
            meta={
                "conversion": "opinion_comment_to_synthetic_rebuttal",
                "is_synthetic": True,
                "synthesis_prompt_version": "v1",
            },
        )
        if record:
            records.append(record)

    return records


if __name__ == "__main__":
    write_jsonl(build_records(), "data/interim/ko_kcbert_comments.jsonl")
```

## Suggested codebase layout

The downstream codebase should only include adapters for datasets that remain enabled or explicitly planned in `configs/data_sources.yaml`. If a Korean dataset is removed from the table above, also remove its adapter file, configuration entry, and implementation-order reference.

```text
.
├── README.md
├── configs/
│   └── data_sources.yaml
├── data/
│   ├── raw/                 # ignored by Git
│   ├── raw_manual/          # ignored by Git; AI Hub/manual downloads
│   ├── interim/             # source-specific JSONL
│   └── processed/           # final train/valid/test JSONL
├── src/
│   ├── preprocess/
│   │   ├── common.py
│   │   ├── en_cmv_convokit.py
│   │   ├── en_ibm_argq.py
│   │   ├── en_args_me.py
│   │   ├── en_perspectrum.py
│   │   ├── en_debatesum.py
│   │   ├── ko_k_news_stance.py
│   │   ├── ko_korean_petitions.py
│   │   ├── ko_aihub_dialogue.py
│   │   ├── ko_kornli.py
│   │   ├── ko_klue_nli.py
│   │   ├── ko_klue_ynat.py
│   │   ├── ko_kcbert_comments.py
│   │   ├── ko_korean_chatbot_data.py
│   │   ├── ko_nsmc.py
│   │   ├── ko_question_pair.py
│   │   ├── ko_kote.py
│   │   ├── ko_korean_hate_speech.py
│   │   ├── ko_kmhas.py
│   │   ├── ko_kold.py
│   │   ├── ko_kopolitic.py
│   │   ├── ko_kornat.py
│   │   ├── ko_popong_politics.py
│   │   ├── ko_modu_news.py
│   │   ├── ko_modu_messenger.py
│   │   ├── ko_open_subtitles.py
│   │   ├── ko_safety_filter.py
│   │   ├── merge_and_split.py
│   │   └── build_vocab.py
│   ├── model/
│   │   └── lstm_seq2seq.py
│   └── train.py
└── requirements.txt
```

## Example configuration

```yaml
# configs/data_sources.yaml

seed: 42

languages:
  - en
  - ko

sampling:
  # Leave null until the final experiment design is decided.
  target_total_per_language: null
  source_weights: null

format:
  add_language_tag: false
  sep_token: "<SEP>"
  sos_token: "<SOS>"
  eos_token: "<EOS>"

filters:
  min_input_chars: 20
  min_target_chars: 20
  max_input_chars: 600
  max_target_chars: 600
  max_encoder_tokens: 120
  max_decoder_tokens: 120
  reject_same_input_target: true
  deduplicate_by_normalized_text: true

sources:
  en:
    cmv_convokit:
      enabled: true
      kind: direct_dialogue
      path: null
    ibm_argq_30k:
      enabled: true
      kind: stance_pair
      hf_name: ibm-research/argument_quality_ranking_30k
    args_me:
      enabled: true
      kind: stance_pair
      hf_name: webis/args_me
    perspectrum:
      enabled: true
      kind: perspective_pair
      path: null
    debatesum:
      enabled: true
      kind: evidence_summary
      hf_name: Hellisotherpeople/DebateSum
    aae2:
      enabled: false
      kind: argument_relation
      hf_name: pie/aae2
    schopenhauer_debate:
      enabled: false
      kind: style_reference
      hf_name: raphaaal/schopenhauer-debate

  ko:
    k_news_stance:
      enabled: true
      kind: stance_pair
      path: data/raw_manual/k-news-stance/k-news-stance.json
    korean_petitions:
      enabled: true
      kind: issue_to_synthetic_rebuttal
      korpora_name: korean_petitions
    aihub_dialogue_summary:
      enabled: true
      kind: direct_dialogue_or_synthetic
      path: data/raw_manual/aihub/dialogue_summary/
    aihub_topic_dialogue:
      enabled: true
      kind: direct_dialogue
      path: data/raw_manual/aihub/topic_dialogue/
    aihub_meeting_speech:
      enabled: false
      kind: transcript_dialogue
      path: data/raw_manual/aihub/meeting_speech/
    aihub_korean_dialogue:
      enabled: false
      kind: auxiliary_dialogue
      path: data/raw_manual/aihub/korean_dialogue/
    kornli:
      enabled: true
      kind: nli_contradiction_pair
      korpora_name: kornli
    klue_nli:
      enabled: true
      kind: nli_contradiction_pair
      hf_name: klue/klue
      hf_config: nli
    klue_ynat:
      enabled: false
      kind: topic_seed
      hf_name: klue/klue
      hf_config: ynat
    kcbert_comments:
      enabled: true
      kind: opinion_comment_to_synthetic_rebuttal
      korpora_name: kcbert
      max_candidates: null
    korean_chatbot_data:
      enabled: false
      kind: auxiliary_dialogue
      korpora_name: korean_chatbot_data
    nsmc:
      enabled: false
      kind: opinion_review_to_synthetic_rebuttal
      korpora_name: nsmc
    question_pair:
      enabled: false
      kind: topic_paraphrase_or_dedup
      korpora_name: question_pair
    kote:
      enabled: false
      kind: emotion_comment_to_synthetic_rebuttal
      hf_name: searle-j/kote
    korean_hate_speech:
      enabled: false
      kind: safety_filter
      korpora_name: korean_hate_speech
    kmhas:
      enabled: false
      kind: safety_filter
      hf_name: jeanlee/kmhas_korean_hate_speech
    kold:
      enabled: false
      kind: safety_filter
      hf_name: nayohan/KOLD
    kopolitic:
      enabled: false
      kind: political_news_to_synthetic_rebuttal
      path: data/raw_manual/kopolitic/
    kornat_social_values:
      enabled: false
      kind: social_value_topic_seed
      hf_name: jiyounglee0523/KorNAT
    popong_politics:
      enabled: false
      kind: political_topic_seed
      path: data/raw_manual/popong-data-for-rnd/
    modu_news:
      enabled: false
      kind: evidence_or_topic_seed
      path: data/raw_manual/nikl/NIKL_NEWSPAPER/
    modu_messenger:
      enabled: false
      kind: auxiliary_dialogue
      path: data/raw_manual/nikl/NIKL_MESSENGER/
    open_subtitles:
      enabled: false
      kind: auxiliary_dialogue_or_translation
      korpora_name: open_substitles
```

## Tokenization notes

For an LSTM seq2seq baseline, keep tokenization simple and reproducible.

Options:

1. **Whitespace/token-level tokenizer** for English and Korean after normalization. Simple but weak for Korean morphology.
2. **Character-level tokenizer** for Korean. Robust against out-of-vocabulary words but produces longer sequences.
3. **SentencePiece unigram/BPE tokenizer** trained on the combined processed corpus. Stronger option, but adds one more dependency and should be documented carefully.

Start with a simple tokenizer, then improve only if training quality is poor. Keep the special tokens fixed:

```text
<PAD>
<UNK>
<SOS>
<EOS>
<SEP>
```

Optional bilingual tokens:

```text
<EN>
<KO>
```

## Quality control checklist

Before training:

- Verify that `encoder_input`, `decoder_input`, and `decoder_target` are non-empty.
- Verify that `<SOS>` appears only at the start of `decoder_input`.
- Verify that `<EOS>` appears only at the end of `decoder_target`.
- Remove duplicate `(topic, input_context, target_output)` triples.
- Split by topic to avoid leakage.
- Inspect random examples from each language manually.
- Check that Korean-native data remains a major part of the Korean subset.
- Keep source attribution and synthetic-generation metadata.
- Do not commit raw datasets whose licenses or access terms prohibit redistribution.

## Initial implementation order

1. Implement `src/preprocess/common.py`.
2. Implement one English stance-pair adapter: `en_ibm_argq.py`.
3. Implement one Korean issue-to-rebuttal adapter: `ko_korean_petitions.py`.
4. Implement selected no-AI-Hub Korean adapters in this order: `ko_kornli.py` or `ko_klue_nli.py`, `ko_kcbert_comments.py`, then `ko_kote.py` or `ko_kopolitic.py`.
5. Implement `ko_safety_filter.py` using Korean Hate Speech Dataset, K-MHaS, or KOLD before large-scale LLM synthesis.
6. Implement `merge_and_split.py` with topic-level splitting.
7. Train the first small LSTM seq2seq baseline.
8. Add ChangeMyView and K-News-Stance adapters.
9. Add AI Hub Korean dialogue adapters only if account/manual access is resolved.
10. Add optional LLM synthesis and mark all generated examples in metadata.

## Known limitations

- Many Korean sources are not directly debate datasets; they require filtering or synthesis.
- Korpora/Hugging Face/GitHub sources avoid AI Hub keys, but some still require separate licenses, manual downloads, or account-based access. Check each source before redistribution.
- NIKL Modu corpora are not AI Hub datasets, but Korpora documentation indicates manual NIKL corpus access is required for some Modu corpora.
- Hate/offensive-language corpora are primarily for filtering and evaluation. They should not be used as positive target outputs unless examples are explicitly clean and manually inspected.
- LLM-synthesized data can become stylistically repetitive. Mix it with source-derived human text and deduplicate aggressively.
- Long evidence documents are unsuitable as direct seq2seq targets. Summarize or truncate them before model training.
- This README lists candidate sources and preprocessing logic only; it does not determine final dataset proportions.
- Keep the dataset table, adapter file list, configuration keys, and implementation order synchronized whenever sources are added or removed.
