"""REL-01: ``agent:end`` says why a turn ended, and every start has an end.

Restored on current upstream after the live gateway lost fork PR #3. The
consumer is the ``turn-exit-monitor`` hook, which reads ``turn_exit_reason``;
without it every abnormal exit reached that hook as an empty string.

The classification tests call the pure helpers. The emission tests run the
real ``_handle_message_with_agent`` with its collaborators stubbed, so what is
asserted is the context the hook actually receives on each exit path.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway import run_turn
from gateway.turn_exit_metadata import (
    GATEWAY_STALE_GENERATION,
    GATEWAY_UNHANDLED_EXCEPTION,
    REASON_MAX_CHARS,
    agent_end_metadata,
    turn_exit_reason,
)


# --------------------------------------------------------------- classification

def test_explicit_finalizer_reason_passes_through():
    assert turn_exit_reason({"turn_exit_reason": "max_iterations_reached(90)"}) == "max_iterations_reached(90)"


def test_missing_reason_on_a_normal_turn_is_unknown():
    assert turn_exit_reason({}) == "unknown"
    assert turn_exit_reason(None) == "unknown"


@pytest.mark.parametrize("marker, expected", [
    ("_INTERRUPT_REASON_TIMEOUT", "gateway_inactivity_timeout"),
    ("_INTERRUPT_REASON_SSE_DISCONNECT", "gateway_sse_disconnect"),
    ("_INTERRUPT_REASON_GATEWAY_SHUTDOWN", "gateway_shutdown"),
    ("_INTERRUPT_REASON_GATEWAY_RESTART", "gateway_restart"),
    ("_INTERRUPT_REASON_EVICTED", "gateway_session_evicted"),
    ("_INTERRUPT_REASON_STOP", "interrupted_by_user"),
    ("_INTERRUPT_REASON_RESET", "interrupted_by_user"),
])
def test_early_interrupt_is_classified_by_its_control_marker(marker, expected):
    result = {"interrupted": True, "interrupt_message": getattr(gateway_run, marker)}
    assert turn_exit_reason(result) == expected


def test_system_marker_overrides_a_generic_agent_interrupt_reason():
    result = {"interrupted": True, "turn_exit_reason": "interrupted",
              "interrupt_message": gateway_run._INTERRUPT_REASON_GATEWAY_SHUTDOWN}
    assert turn_exit_reason(result) == "gateway_shutdown"


def test_every_control_marker_has_an_actor():
    """A new control marker must not fall through to 'unclassified'."""
    for marker in gateway_run._CONTROL_INTERRUPT_MESSAGES:
        reason = turn_exit_reason({"interrupted": True, "interrupt_message": marker})
        assert reason != "gateway_interrupt_unclassified", marker
        assert reason == "interrupted_by_user" or reason.startswith("gateway_"), (marker, reason)


def test_mid_turn_correction_text_is_a_user_interrupt():
    assert turn_exit_reason({"interrupted": True, "interrupt_message": "아니 그거 말고"}) == "interrupted_by_user"


def test_interrupt_without_marker_or_reason_stays_unclassified():
    assert turn_exit_reason({"interrupted": True}) == "gateway_interrupt_unclassified"


def test_unknown_reason_is_not_exported_as_provider_text():
    meta = agent_end_metadata({"turn_exit_reason": "a\nb\x07c " + "x" * 500})
    assert meta["turn_exit_reason"] == "unknown"
    assert len(meta["turn_exit_reason"]) <= REASON_MAX_CHARS


class _Unstringable:
    def __str__(self):
        raise RuntimeError("no")


class _ExplodingCount:
    def __int__(self):
        raise RuntimeError("no")

    def __bool__(self):
        return True


@pytest.mark.parametrize("api_calls, expected", [(7, 7), (-3, 0), ("x", 0), (None, 0), (_ExplodingCount(), 0)])
def test_count_is_non_negative_and_fail_safe(api_calls, expected):
    assert agent_end_metadata({"api_calls": api_calls})["api_call_count"] == expected


def test_malformed_reason_fails_safe():
    assert agent_end_metadata({"turn_exit_reason": _Unstringable()})["turn_exit_reason"] == "unknown"


def test_stale_normal_finish_says_why_it_was_dropped():
    meta = agent_end_metadata({"turn_exit_reason": "text_response(finish_reason=stop)"}, stale=True)
    assert meta == {"turn_exit_reason": GATEWAY_STALE_GENERATION, "api_call_count": 0,
                    "api_call_count_kind": "legacy_iteration", "stale": True}


def test_stale_abnormal_reason_is_preserved():
    meta = agent_end_metadata({"turn_exit_reason": "max_iterations_reached(90)", "api_calls": 90}, stale=True)
    assert meta["turn_exit_reason"] == "max_iterations_reached(90)" and meta["stale"] is True


# ------------------------------------------------------------------- emission

@dataclass
class _Prepared:
    history: Any = ()
    context_prompt: str = ""
    message_text: Any = "hello"
    persist_user_message: Any = None
    persist_user_timestamp: Any = None
    persist_user_display_kind: Optional[str] = None
    persistence_session_id: Optional[str] = None
    persistence_owner: Optional[str] = None
    title_user_message: Optional[str] = None


def _runner(monkeypatch, *, agent_result=None, run_raises=None, current=True):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(run_turn, "persisted_user_metadata_for_event", lambda event: {})
    monkeypatch.setattr(run_turn, "reply_expected_metadata", lambda value: {})
    monkeypatch.setattr(run_turn, "diagnostic_metadata", lambda event: {})
    import gateway.run_heartbeat_acceptance as hb
    monkeypatch.setattr(hb, "heartbeat_owner_is_current", lambda *a, **k: True)

    source = types.SimpleNamespace(platform=types.SimpleNamespace(value="discord"), user_id="u",
                                   chat_id="c", thread_id="t", chat_type="thread", user_name="n")
    entry = types.SimpleNamespace(session_id="s1")
    r = MagicMock()
    r._PreparedTurn = _Prepared
    r.hooks.emit = AsyncMock()
    r._hmwa_resolve_session = AsyncMock(return_value=(source, entry, "key"))
    r._hmwa_prepare_turn = AsyncMock(return_value=(_Prepared(), None))
    r._pinned_channel_inputs = MagicMock(return_value=(None, source))
    r._persist_prompt_pins = AsyncMock()
    r._run_agent = AsyncMock(side_effect=run_raises, return_value=agent_result)
    r._hmwa_stop_typing_for_turn = AsyncMock()
    r._is_session_run_current = MagicMock(return_value=current)
    r._hmwa_shape_agent_response = AsyncMock(return_value=("reply", False, []))
    r._usage_window_chat_wording = lambda result, response: response
    r._hmwa_prepend_reasoning = lambda result, response, source, silence: response
    r._hmwa_runtime_footer_line = MagicMock(return_value="")
    r._hmwa_post_turn_hooks = types.MethodType(GatewayRunner._hmwa_post_turn_hooks, r)
    r._drain_watch_notifications = AsyncMock()
    r._hmwa_classify_turn_failure = MagicMock(return_value=(False, False, False))
    r._hmwa_compression_exhaustion_reset = AsyncMock(return_value=("reply", entry))
    r._hmwa_persist_turn_transcript = AsyncMock()
    r._record_room_silence_decision = AsyncMock()
    r._deliver_turn_response_then_limit_offer = AsyncMock(return_value="delivered")
    r._hmwa_agent_error_reply = AsyncMock(return_value="error-reply")
    event = MagicMock(internal=False, reply_expected=None, metadata={}, message_id="m1", text="hello")
    run = types.MethodType(GatewayRunner._handle_message_with_agent, r)
    return r, event, source, run


def _agent_end_calls(r):
    return [c.args[1] for c in r.hooks.emit.await_args_list if c.args[0] == "agent:end"]


@pytest.mark.asyncio
async def test_normal_turn_reports_reason_and_count(monkeypatch):
    r, event, source, run = _runner(monkeypatch, agent_result={
        "turn_exit_reason": "max_iterations_reached(90)", "api_calls": 90, "model": "m", "provider": "p"})
    await run(event, source, "key", 1)
    (ctx,) = _agent_end_calls(r)
    assert ctx["turn_exit_reason"] == "max_iterations_reached(90)"
    assert ctx["api_call_count"] == 90 and ctx["stale"] is False
    assert ctx["response"] == "reply" and ctx["session_id"] == "s1"


@pytest.mark.asyncio
async def test_superseded_turn_still_ends_and_says_stale(monkeypatch):
    r, event, source, run = _runner(monkeypatch, current=False, agent_result={
        "turn_exit_reason": "text_response(finish_reason=stop)", "api_calls": 2})
    assert await run(event, source, "key", 1) is None
    (ctx,) = _agent_end_calls(r)
    assert ctx["stale"] is True and ctx["response"] == ""
    assert ctx["turn_exit_reason"] == GATEWAY_STALE_GENERATION
    r._hmwa_discard_stale_result.assert_called_once()


@pytest.mark.asyncio
async def test_crash_emits_exactly_one_agent_end_and_keeps_the_error_path(monkeypatch):
    r, event, source, run = _runner(monkeypatch, run_raises=RuntimeError("boom"))
    assert await run(event, source, "key", 1) == "error-reply"
    (ctx,) = _agent_end_calls(r)
    assert ctx["turn_exit_reason"] == GATEWAY_UNHANDLED_EXCEPTION
    assert ctx["api_call_count"] == 0 and ctx["stale"] is False
    r._hmwa_agent_error_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_crash_after_the_normal_end_does_not_double_emit(monkeypatch):
    r, event, source, run = _runner(monkeypatch, agent_result={"turn_exit_reason": "x", "api_calls": 1})
    r._hmwa_persist_turn_transcript = AsyncMock(side_effect=RuntimeError("late"))
    await run(event, source, "key", 1)
    assert len(_agent_end_calls(r)) == 1


@pytest.mark.asyncio
async def test_a_failing_crash_emit_does_not_swallow_the_error_reply(monkeypatch):
    r, event, source, run = _runner(monkeypatch, run_raises=RuntimeError("boom"))
    real_emit = AsyncMock()

    async def emit(name, ctx):
        await real_emit(name, ctx)
        if name == "agent:end":
            raise RuntimeError("hook bus down")
    r.hooks.emit = emit
    assert await run(event, source, "key", 1) == "error-reply"
