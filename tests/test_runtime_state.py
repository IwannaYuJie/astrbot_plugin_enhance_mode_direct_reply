from __future__ import annotations

from astrbot_plugin_enhance_mode_direct_reply.runtime_state import RuntimeState


def test_touch_origin_evicts_oldest_state() -> None:
    state = RuntimeState()

    state.session_chats["o1"].append("m1")
    state.active_reply_stacks["o1"].append("a1")
    state.model_choice_histories["o1"].append("h1")
    state.active_reply_send_timestamps["o1"].append(1.0)
    state.image_message_registry["o1"]["mid"] = {"urls": ["u1"], "captions": {}}
    state.pending_active_reply_jobs["o1"] = {"pending_id": "p1"}

    state.touch_origin("o1", max_origins=1)
    state.touch_origin("o2", max_origins=1)

    assert "o1" not in state.session_chats
    assert "o1" not in state.active_reply_stacks
    assert "o1" not in state.model_choice_histories
    assert "o1" not in state.active_reply_send_timestamps
    assert "o1" not in state.image_message_registry
    assert "o1" not in state.pending_active_reply_jobs
    assert list(state.origin_lru.keys()) == ["o2"]


def test_cleanup_origin_removes_all_runtime_state() -> None:
    state = RuntimeState()
    state.session_chats["origin"].append("msg")
    state.active_reply_stacks["origin"].append("stack")
    state.model_choice_histories["origin"].append("hist")
    state.active_reply_send_timestamps["origin"].append(1.0)
    state.image_message_registry["origin"]["1"] = {"urls": ["x"], "captions": {}}
    state.pending_active_reply_jobs["origin"] = {"pending_id": "p1"}
    state.touch_origin("origin", max_origins=10)

    state.cleanup_origin("origin")

    assert "origin" not in state.session_chats
    assert "origin" not in state.active_reply_stacks
    assert "origin" not in state.model_choice_histories
    assert "origin" not in state.active_reply_send_timestamps
    assert "origin" not in state.image_message_registry
    assert "origin" not in state.pending_active_reply_jobs
    assert "origin" not in state.origin_lru
