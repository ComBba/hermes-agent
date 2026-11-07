import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from gateway.turn_exit_metadata import agent_end_metadata
from tests.agent.test_codex_app_server_integration import _make_codex_agent
from tests.agent.test_retry_exhaustion_partial_retention import _Agent, _Http
from tests.agent.test_outer_error_retry_bound import _PermanentError, loop_agent
from tests.gateway.test_agent_end_turn_exit_metadata import _agent_end_calls, _runner


@pytest.mark.asyncio
@pytest.mark.parametrize("error,status,calls,expected", [
    (_Http(401, "HTTP 401: invalid api key"), 401, 1, "api_auth"),
    (_Http(403, "HTTP 403: invalid api key"), 403, 3, "api_auth"),
    (_Http(429, "HTTP 429: upstream failure"), 429, 5, "api_rate_limit"),
    (_Http(500, "HTTP 500: upstream failure"), 500, 3, "api_server_error"),
    (_Http(503, "HTTP 503: upstream failure"), 503, 1, "api_overloaded"),
    (TimeoutError("read timed out"), None, 3, "api_timeout"),
    (ConnectionError("connection reset"), None, 5, "api_timeout"),
])
async def test_real_terminal_error_builder_reaches_agent_end_with_exact_call_count(monkeypatch, error, status, calls, expected):
    classified = classify_api_error(error, provider="openrouter", model="m")
    retry = SimpleNamespace(provider_stale_refresh_attempted=False, primary_recovery_attempted=True)
    with (
        patch("agent.conversation_loop._credential_refresh_profile", lambda agent: None),
        patch("agent.turn_recovery_autorecover.auto_recover_after_exhaustion", lambda *a, **kw: None),
    ):
        verdict = settle_unrecovered_error(
            _Agent(), api_error=error, classified=classified, _retry=retry, status_code=status,
            error_msg=str(error).lower(), is_context_length_error=False, is_rate_limited=status == 429,
            _is_zai_coding_overload=False, _provider="openrouter", _base="https://fixture.invalid/v1",
            _model="m", messages=[{"role": "user", "content": "fixture"}], api_messages=[], api_kwargs=None,
            active_system_prompt="", conversation_history=None, approx_tokens=10, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=calls, current_turn_user_idx=0,
        )
    assert verdict.action == "return"
    result = json.loads(json.dumps(verdict.result))
    runner, event, source, execute = _runner(monkeypatch, agent_result=result)
    assert await execute(event, source, "key", 1) == "delivered"
    (context,) = _agent_end_calls(runner)
    assert context["turn_exit_reason"] == expected
    assert context["api_call_count"] == calls


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "context_compression_timeout", "context_compression_exhausted", "session_persistence_failed",
    "pending_tool_result", "empty_response_exhausted", "interrupted_during_api_call",
])
async def test_terminal_runtime_reason_survives_actual_gateway_persistence_hook(monkeypatch, reason):
    runner, event, source, execute = _runner(monkeypatch, agent_result={"turn_exit_reason": reason, "api_calls": 3})
    await execute(event, source, "key", 1)
    (context,) = _agent_end_calls(runner)
    assert (context["turn_exit_reason"], context["api_call_count"], context["stale"]) == (reason, 3, False)


@pytest.mark.parametrize("kind,expected,calls", [
    ("start_error", "local_processing_error", 0), ("run_error", "local_processing_error", 1),
    ("turn_error", "local_processing_error", 1), ("interrupted", "interrupted_during_api_call", 1),
    ("empty", "empty_response_exhausted", 1), ("normal", "text_response(finish_reason=stop)", 1),
])
def test_codex_runtime_produces_typed_termination_and_attempt_count(monkeypatch, kind, expected, calls):
    def start(session):
        if kind == "start_error":
            raise RuntimeError("PRIVATE-START-ERROR")
        return "fixture-thread"

    def run(session, user_input, **kwargs):
        if kind == "run_error":
            raise RuntimeError("PRIVATE-RUN-ERROR")
        text = "done" if kind == "normal" else ""
        return TurnResult(final_text=text, projected_messages=[{"role": "assistant", "content": text}] if text else [],
                          interrupted=kind == "interrupted", error="PRIVATE-TURN-ERROR" if kind == "turn_error" else None,
                          thread_id="fixture-thread", turn_id="fixture-turn")

    monkeypatch.setattr(CodexAppServerSession, "ensure_started", start)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", run)
    agent = _make_codex_agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("fixture")
    metadata = agent_end_metadata(result)
    assert metadata["turn_exit_reason"] == expected
    assert metadata["api_call_count"] == calls
    assert metadata["api_call_count_kind"] == "codex_turn_attempt"
    assert "PRIVATE" not in json.dumps(metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [0, 2, 4])
async def test_real_loop_observed_attempts_survive_gateway_reporting(monkeypatch, loop_agent, failures):
    messages = [_PermanentError("fixture-assembly-error")] * failures
    messages.append({"role": "assistant", "content": "ok", "finish_reason": "stop"})
    with (
        patch.object(loop_agent, "_build_assistant_message", side_effect=messages),
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("fixture")
    attempts = loop_agent.client.chat.completions.create.call_count
    assert attempts == failures + 1
    assert result["api_calls"] == attempts
    runner, event, source, execute = _runner(monkeypatch, agent_result=result)
    await execute(event, source, "key", 1)
    (context,) = _agent_end_calls(runner)
    assert context["api_call_count"] == attempts
    assert context["turn_exit_reason"] == "text_response(finish_reason=stop)"
