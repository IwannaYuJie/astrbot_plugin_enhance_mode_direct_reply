from __future__ import annotations

from astrbot.api.message_components import At, Plain

from astrbot_plugin_enhance_mode_direct_reply.tag_utils import (
    build_interaction_instructions,
    clean_response_text_for_history,
    transform_result_chain,
)


def test_transform_result_chain_strips_quote_tags_without_reply_component() -> None:
    transformed = transform_result_chain(
        [Plain(text='<quote id="12345"/> hello there')],
        parse_mention=True,
    )

    assert transformed is not None
    assert len(transformed) == 1
    assert isinstance(transformed[0], Plain)
    assert transformed[0].text == " hello there"


def test_transform_result_chain_still_parses_mentions() -> None:
    transformed = transform_result_chain(
        [Plain(text='<quote id="12345"/><mention id="42"/> hi')],
        parse_mention=True,
    )

    assert transformed is not None
    assert len(transformed) == 2
    assert isinstance(transformed[0], At)
    assert transformed[0].qq == "42"
    assert isinstance(transformed[1], Plain)
    assert transformed[1].text == " hi"


def test_build_interaction_instructions_forbids_quote_tags() -> None:
    instructions = build_interaction_instructions(
        mention_parse=True,
        include_sender_id=True,
    )

    assert "Reply with a normal message only." in instructions
    assert "Do NOT quote or reference a specific message" in instructions
    assert "When you want to quote/reply to a specific message" not in instructions


def test_clean_response_text_for_history_removes_quote_tags() -> None:
    cleaned = clean_response_text_for_history('<quote id="12345"/>test')

    assert cleaned == "test"
