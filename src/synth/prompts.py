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
