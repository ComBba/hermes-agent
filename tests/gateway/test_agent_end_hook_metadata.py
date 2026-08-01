"""Gateway agent:end hook termination metadata."""

import sys
import threading
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _event():
    return MessageEvent(
        text="run a task",
        source=_source(),
        message_id="msg-42",
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-agent-end",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_exit_reason", "api_calls", "interrupted"),
    [
        ("text_response(finish_reason=stop)", 1, False),
        ("max_iterations_reached(90/90)", 90, False),
        ("guardrail_halt", 3, False),
        ("all_retries_exhausted_no_response", 3, False),
        ("interrupted_by_user", 2, True),
    ],
)
async def test_agent_end_hook_includes_termination_metadata(
    monkeypatch,
    tmp_path,
    turn_exit_reason,
    api_calls,
    interrupted,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "done",
            "messages": [
                {"role": "user", "content": "run a task"},
                {"role": "assistant", "content": "done"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": api_calls,
            "turn_exit_reason": turn_exit_reason,
            "failed": False,
            "interrupted": interrupted,
            "interrupt_message": (
                gateway_run._INTERRUPT_REASON_STOP if interrupted else None
            ),
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end_contexts = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ]
    assert len(agent_end_contexts) == 1
    assert agent_end_contexts[0]["turn_exit_reason"] == turn_exit_reason
    assert agent_end_contexts[0]["api_call_count"] == api_calls
    assert agent_end_contexts[0]["stale"] is False
    assert isinstance(agent_end_contexts[0]["turn_exit_reason"], str)
    assert isinstance(agent_end_contexts[0]["api_call_count"], int)
    assert agent_end_contexts[0]["session_id"] == "sess-agent-end"
    assert agent_end_contexts[0]["user_id"] == "12345"


@pytest.mark.asyncio
async def test_agent_end_hook_reports_gateway_unhandled_exception_once(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        side_effect=RuntimeError("synthetic unhandled agent failure")
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end_contexts = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ]
    assert len(agent_end_contexts) == 1
    assert (
        agent_end_contexts[0]["turn_exit_reason"]
        == "gateway_unhandled_exception"
    )
    assert agent_end_contexts[0]["api_call_count"] == 0
    assert agent_end_contexts[0]["stale"] is False


@pytest.mark.asyncio
async def test_agent_end_hook_normalizes_early_user_interrupt(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Operation interrupted during retry.",
            "messages": [],
            "tools": [],
            "api_calls": 2,
            "interrupted": True,
            "interrupt_message": gateway_run._INTERRUPT_REASON_STOP,
            "completed": False,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == "interrupted_by_user"
    assert agent_end["api_call_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupt_message", "expected_reason"),
    [
        (gateway_run._INTERRUPT_REASON_STOP, "interrupted_by_user"),
        (gateway_run._INTERRUPT_REASON_RESET, "interrupted_by_user"),
        (
            gateway_run._INTERRUPT_REASON_TIMEOUT,
            "gateway_inactivity_timeout",
        ),
        (
            gateway_run._INTERRUPT_REASON_SSE_DISCONNECT,
            "gateway_sse_disconnect",
        ),
        (
            gateway_run._INTERRUPT_REASON_GATEWAY_SHUTDOWN,
            "gateway_shutdown",
        ),
        (
            gateway_run._INTERRUPT_REASON_GATEWAY_RESTART,
            "gateway_restart",
        ),
        (None, "gateway_interrupt_unclassified"),
        ("new correction", "interrupted_by_user"),
    ],
)
async def test_agent_end_hook_classifies_early_interrupt_actor(
    monkeypatch,
    tmp_path,
    interrupt_message,
    expected_reason,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Operation interrupted.",
            "messages": [],
            "tools": [],
            "api_calls": 1,
            "interrupted": True,
            "interrupt_message": interrupt_message,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == expected_reason


def test_control_interrupt_markers_have_exhaustive_actor_classification():
    classified = (
        set(gateway_run._SYSTEM_INTERRUPT_EXIT_REASONS)
        | gateway_run._USER_INTERRUPT_MESSAGES
    )
    assert classified == gateway_run._CONTROL_INTERRUPT_MESSAGES


@pytest.mark.asyncio
async def test_agent_end_hook_preserves_explicit_reason_over_interrupt_fallback(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "halted",
            "messages": [],
            "tools": [],
            "api_calls": 1,
            "interrupted": True,
            "interrupt_message": gateway_run._INTERRUPT_REASON_TIMEOUT,
            "turn_exit_reason": " guardrail_halt\x00\x1b[31m\n\twith context ",
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == "guardrail_halt[31m with context"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generic_reason", "system_marker", "expected_reason"),
    [
        (
            "Interrupted_By_User(iteration=7)",
            gateway_run._INTERRUPT_REASON_SSE_DISCONNECT,
            "gateway_sse_disconnect",
        ),
        (
            "interrupted_during_api_call",
            gateway_run._INTERRUPT_REASON_GATEWAY_SHUTDOWN,
            "gateway_shutdown",
        ),
    ],
)
async def test_agent_end_hook_system_marker_overrides_generic_agent_interrupt(
    monkeypatch,
    tmp_path,
    generic_reason,
    system_marker,
    expected_reason,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "interrupted",
            "messages": [],
            "tools": [],
            "api_calls": 1,
            "interrupted": True,
            "interrupt_message": system_marker,
            "turn_exit_reason": generic_reason,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == expected_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_result", "expected_reason"),
    [
        (
            {"turn_exit_reason": "interrupted_by_user"},
            "interrupted_by_user",
        ),
        (
            {
                "turn_exit_reason": "interrupted_during_api_call",
                "interrupted": True,
                "interrupt_message": "future gateway interrupt",
            },
            "interrupted_during_api_call",
        ),
    ],
)
async def test_agent_end_hook_preserves_generic_reason_without_known_marker(
    monkeypatch,
    tmp_path,
    agent_result,
    expected_reason,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "interrupted",
            "messages": [],
            "tools": [],
            "api_calls": 1,
            **agent_result,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == expected_reason


class _UnstringableReason:
    def __bool__(self):
        raise TypeError("malformed reason")

    def __str__(self):
        raise TypeError("malformed reason")


class _ExplodingCount:
    def __bool__(self):
        raise RuntimeError("malformed count")


@pytest.mark.asyncio
async def test_agent_end_hook_bounds_reason_and_fails_safe_on_bad_count(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "failed",
            "messages": [],
            "tools": [],
            "api_calls": object(),
            "turn_exit_reason": "local_processing_error(" + ("x" * 500) + ")",
            "failed": True,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert len(agent_end["turn_exit_reason"]) == 200
    assert agent_end["turn_exit_reason"].startswith("local_processing_error(")
    assert agent_end["api_call_count"] == 0


@pytest.mark.asyncio
async def test_agent_end_hook_still_emits_when_count_protocol_raises(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "failed",
            "messages": [],
            "tools": [],
            "api_calls": _ExplodingCount(),
            "turn_exit_reason": "guardrail_halt",
            "failed": True,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == "guardrail_halt"
    assert agent_end["api_call_count"] == 0


@pytest.mark.asyncio
async def test_agent_end_hook_fails_safe_on_malformed_reason(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "failed",
            "messages": [],
            "tools": [],
            "api_calls": 1,
            "turn_exit_reason": _UnstringableReason(),
            "failed": True,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == "unknown"


@pytest.mark.asyncio
async def test_agent_end_hook_clamps_negative_count(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "done",
            "messages": [],
            "tools": [],
            "api_calls": -3,
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "failed": False,
        }
    )

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["api_call_count"] == 0


@pytest.mark.asyncio
async def test_stale_proxy_result_emits_agent_end_before_discard(
    monkeypatch,
    tmp_path,
):
    events = []

    class _StaleAdapter:
        def pop_post_delivery_callback(self, _key, *, generation):
            assert generation == 1
            events.append("cleanup")

        async def send(self, _chat_id, _content, *, metadata):
            return SimpleNamespace(success=True)

    runner = _runner(monkeypatch, tmp_path)
    runner._is_session_run_current = lambda _key, _gen: False
    stale_adapter = _StaleAdapter()
    runner._adapter_for_source = lambda _source: stale_adapter

    async def emit_after_cleanup(event_type, _context):
        if event_type == "agent:end":
            events.append("emit")

    runner.hooks.emit.side_effect = emit_after_cleanup
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "",
            "messages": [],
            "tools": [],
            "api_calls": 0,
            "partial": False,
            "turn_exit_reason": "gateway_proxy_stale_generation",
        }
    )

    result = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert result is None
    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ]
    assert len(agent_end) == 1
    assert agent_end[0]["turn_exit_reason"] == "gateway_proxy_stale_generation"
    assert agent_end[0]["api_call_count"] == 0
    assert agent_end[0]["response"] == ""
    assert agent_end[0]["stale"] is True
    assert events == ["cleanup", "emit"]


@pytest.mark.asyncio
async def test_stale_non_proxy_result_pairs_agent_start_and_end(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._is_session_run_current = lambda _key, _gen: False
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "discarded",
            "messages": [],
            "tools": [],
            "api_calls": 1,
            "partial": False,
            "turn_exit_reason": "text_response(finish_reason=stop)",
        }
    )

    result = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert result is None
    lifecycle_events = [
        (call.args[0], call.args[1])
        for call in runner.hooks.emit.await_args_list
        if call.args[0] in {"agent:start", "agent:end"}
    ]
    assert [event_type for event_type, _context in lifecycle_events] == [
        "agent:start",
        "agent:end",
    ]
    assert (
        lifecycle_events[1][1]["turn_exit_reason"]
        == "gateway_stale_generation"
    )
    assert lifecycle_events[1][1]["response"] == ""
    assert lifecycle_events[1][1]["stale"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_result", "expected_reason"),
    [
        (
            {
                "final_response": "",
                "messages": [],
                "tools": [],
                "api_calls": 0,
            },
            "gateway_stale_generation",
        ),
        (
            {
                "final_response": "proxy output",
                "messages": [],
                "tools": [],
                "api_calls": 1,
                "turn_exit_reason": "gateway_proxy_response_complete",
            },
            "gateway_proxy_stale_generation",
        ),
        (
            {
                "final_response": "",
                "messages": [],
                "tools": [],
                "api_calls": 1,
                "turn_exit_reason": "gateway_inactivity_timeout",
            },
            "gateway_inactivity_timeout",
        ),
    ],
)
async def test_stale_agent_end_preserves_provenance_and_abnormal_reason(
    monkeypatch,
    tmp_path,
    agent_result,
    expected_reason,
):
    runner = _runner(monkeypatch, tmp_path)
    runner._is_session_run_current = lambda _key, _gen: False
    runner._run_agent = AsyncMock(return_value=agent_result)

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    agent_end = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    ][0]
    assert agent_end["turn_exit_reason"] == expected_reason
    assert agent_end["response"] == ""
    assert agent_end["stale"] is True


def _runtime_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=None,
    )
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_response", "turn_exit_reason", "completed", "failed"),
    [
        ("done", "text_response(finish_reason=stop)", True, False),
        ("halted", "guardrail_halt", False, True),
        (None, "all_retries_exhausted_no_response", False, True),
    ],
)
async def test_run_agent_propagates_exit_reason_through_result_mapping(
    monkeypatch,
    tmp_path,
    final_response,
    turn_exit_reason,
    completed,
    failed,
):
    class _TerminationMetadataAgent:
        def __init__(self, **_kwargs):
            self.tools = []
            self._interrupt_requested = False

        @property
        def is_interrupted(self):
            return self._interrupt_requested

        def run_conversation(
            self,
            _message,
            conversation_history=None,
            task_id=None,
            **_kwargs,
        ):
            return {
                "final_response": final_response,
                "messages": [],
                "api_calls": 3,
                "completed": completed,
                "failed": failed,
                "error": "synthetic failure" if failed else None,
                "turn_exit_reason": turn_exit_reason,
            }

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _TerminationMetadataAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )

    result = await _runtime_runner()._run_agent(
        message="run a task",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-agent-end-runtime",
        session_key="agent:main:telegram:group:-1001:12345",
    )

    assert result["turn_exit_reason"] == turn_exit_reason
    assert result["api_calls"] == 3
    assert result["completed"] is completed
    assert result["failed"] is failed


@pytest.mark.asyncio
async def test_run_agent_classifies_runtime_resolution_failure(
    monkeypatch,
    tmp_path,
):
    runner = _runtime_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=RuntimeError("synthetic runtime resolution failure")
    )

    result = await runner._run_agent(
        message="run a task",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-agent-end-auth-failure",
        session_key="agent:main:telegram:group:-1001:12345",
    )

    assert result["failed"] is True
    assert result["turn_exit_reason"] == "gateway_agent_runtime_resolution_failed"


@pytest.mark.asyncio
async def test_run_agent_classifies_inactivity_timeout(monkeypatch, tmp_path):
    interrupted = threading.Event()

    class _InactiveAgent:
        def __init__(self, **_kwargs):
            self.tools = []
            self._interrupt_requested = False

        @property
        def is_interrupted(self):
            return self._interrupt_requested

        def run_conversation(
            self,
            _message,
            conversation_history=None,
            task_id=None,
            **_kwargs,
        ):
            interrupted.wait(timeout=2)
            return {
                "final_response": None,
                "messages": [],
                "api_calls": 1,
                "completed": False,
                "failed": True,
            }

        def get_activity_summary(self):
            return {
                "last_activity_desc": "synthetic stalled provider",
                "seconds_since_activity": 60,
                "current_tool": None,
                "api_call_count": 1,
                "max_iterations": 90,
            }

        def interrupt(self, _reason):
            self._interrupt_requested = True
            interrupted.set()

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _InactiveAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0.001")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0")
    monkeypatch.setattr(gateway_run, "_GATEWAY_AGENT_POLL_INTERVAL_SECONDS", 0.01)

    result = await _runtime_runner()._run_agent(
        message="run a task",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-agent-end-inactivity",
        session_key="agent:main:telegram:group:-1001:12345",
    )

    assert result["failed"] is True
    assert result["turn_exit_reason"] == "gateway_inactivity_timeout"
    assert interrupted.is_set()
