"""Synthesis prompt templates. Bump the version when you change the wording."""

from __future__ import annotations

COUNTERARGUMENT_PROMPT_VERSION = "v1"

COUNTERARGUMENT_PROMPT = """\
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
"""


def render_counterargument_prompt(
    *, language: str, topic: str, input_context: str
) -> str:
    return COUNTERARGUMENT_PROMPT.format(
        language=language, topic=topic, input_context=input_context
    )


# ---------------------------------------------------------------------------
# Full debate pair generation — for topic-seeded synthesis when no source
# argument exists (e.g. just a curated motion list).
# ---------------------------------------------------------------------------

DEBATE_PAIR_PROMPT_VERSION = "v1"

DEBATE_PAIR_PROMPT = """\
You are generating training data for a debate chatbot.
Language: {language}
Topic (motion): {topic}

Write one short PRO argument (1-2 sentences) and one short CON argument
(1-2 sentences) for this motion. The CON must directly address the PRO.

Output exactly two lines, in this format:
PRO: <argument>
CON: <argument>

Requirements:
- Use natural {language}, formal register.
- Each side should make a clear, substantive argument.
- The CON must respond to the PRO, not just present an opposing view.
- Do not invent specific statistics, names, dates, or legal citations.
- Output only the two lines starting with PRO: and CON: — no preamble or commentary.
"""


def render_debate_pair_prompt(*, language: str, topic: str) -> str:
    return DEBATE_PAIR_PROMPT.format(language=language, topic=topic)


# ---------------------------------------------------------------------------
# Domain-seeded debate generation — caller supplies a broad domain (e.g.
# "교육", "사회이슈", "자연/환경") and the LLM proposes a specific motion
# within that domain plus a PRO/CON exchange. Used by the AI-Hub topic
# adapter so each call yields a unique motion grounded in real-world domains.
# ---------------------------------------------------------------------------

DOMAIN_DEBATE_PROMPT_VERSION = "v1"

DOMAIN_DEBATE_PROMPT = """\
You are generating training data for a debate chatbot.
Language: {language}
Domain: {domain}

Propose ONE specific, contestable debate motion within this domain, then write
a short PRO argument (1-2 sentences) and a short CON argument (1-2 sentences).
The CON must directly address the PRO.

Output exactly three lines, in this format:
MOTION: <a single concrete debate motion, in {language}>
PRO: <argument, in {language}>
CON: <argument, in {language}>

Requirements:
- The motion must be a clear, debatable proposition — not a question, not a category.
- Vary the motion across calls; avoid generic or overly broad phrasing.
- Use natural {language}, formal register.
- The CON must respond to the PRO, not just state an opposing view.
- Do not invent specific statistics, names, dates, or legal citations.
- Output only the three lines starting with MOTION:, PRO:, CON: — no preamble or commentary.
"""


def render_domain_debate_prompt(*, language: str, domain: str) -> str:
    return DOMAIN_DEBATE_PROMPT.format(language=language, domain=domain)
