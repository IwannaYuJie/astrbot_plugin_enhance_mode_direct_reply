from __future__ import annotations

import json

import pytest

from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    GlobalSettingsConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


class _DummyEvent:
    def __init__(self, origin: str) -> None:
        self.unified_msg_origin = origin


@pytest.mark.asyncio
async def test_use_image_attach_only_works_without_caption_enabled() -> None:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()

    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=True, image_caption=False),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        global_settings=GlobalSettingsConfig(),
    )
    plugin._cfg = lambda: cfg

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_get_image_caption should not be called in attach-only mode")

    async def resolve_local_path(_image_ref: str) -> str:
        return "/tmp/fake-image.png"

    plugin._get_image_caption = should_not_be_called
    plugin._resolve_image_ref_to_local_path = resolve_local_path
    plugin._encode_image_file = lambda _path: ("ZmFrZQ==", "image/png")

    event = _DummyEvent("origin-1")
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
        prompt="ignored",
    ):
        results.append(item)

    assert len(results) == 2
    payload_text = results[-1].content[0].text
    payload = json.loads(payload_text)
    assert payload["success"] is True
    assert payload["attach_requested"] is True
    assert payload["attach_success"] is True
    assert payload["write_to_history_requested"] is False
