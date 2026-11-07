from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import httpx

from agent.chat_completion_headers import _chat_completion_capturing_headers
from agent.observed_client_invocations import observe_client_invocations, record_client_invocation
from gateway.turn_exit_metadata import agent_end_metadata, crash_metadata
from tests.agent.test_outer_error_retry_bound import loop_agent
from tests.agent.test_retry_exhaustion_partial_retention import _Http
from tests.agent.test_run_agent import _mock_response


def _agent(mode="chat_completions", provider="fixture"):
    return SimpleNamespace(api_mode=mode, provider=provider)


def test_counter_resets_between_turns_and_after_exception():
    with pytest.raises(RuntimeError), observe_client_invocations(_agent()) as first:
        record_client_invocation()
        raise RuntimeError("fixture")
    record_client_invocation()
    assert first is not None and first.snapshot()["count"] == 1
    with observe_client_invocations(_agent()) as second:
        assert second is not None and second.snapshot()["count"] == 0
        record_client_invocation()
        assert second.snapshot()["count"] == 1
    assert crash_metadata()["api_call_count_kind"] == "unavailable"


def test_copied_context_workers_share_only_their_turn_counter():
    barrier = Barrier(2)

    def turn():
        with observe_client_invocations(_agent()) as counter:
            context = copy_context()
            barrier.wait(timeout=5)
            context.run(lambda: [record_client_invocation() for _ in range(25)])
            assert counter is not None
            return counter.snapshot()["count"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        one, two = pool.submit(turn), pool.submit(turn)
        assert (one.result(timeout=5), two.result(timeout=5)) == (25, 25)


def test_late_worker_cannot_increment_the_next_turn():
    release = Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with observe_client_invocations(_agent()) as old:
            context = copy_context()

            def late():
                assert release.wait(timeout=5)
                record_client_invocation()

            future = pool.submit(context.run, late)
        with observe_client_invocations(_agent()) as current:
            record_client_invocation()
            release.set()
            future.result(timeout=5)
            assert current is not None and current.snapshot()["count"] == 1
        assert old is not None and old.snapshot()["count"] == 1


@pytest.mark.parametrize("mode,provider", [("anthropic_messages", "fixture"), ("codex_app_server", "fixture"),
                                          ("chat_completions", "moa")])
def test_uninstrumented_backend_is_never_labeled_observed(mode, provider):
    with observe_client_invocations(_agent(mode, provider)) as counter:
        record_client_invocation()
        assert counter is None


@pytest.mark.parametrize("branch", ["plain", "window-without-raw", "raw"])
def test_each_exclusive_header_client_branch_records_exactly_one_invocation(branch):
    agent = _agent(provider="nous" if branch != "plain" else "fixture")
    agent._credits_state = SimpleNamespace(is_usage_window=True)
    create = MagicMock(return_value="reply")
    completions = SimpleNamespace(create=create)
    if branch == "raw":
        raw = SimpleNamespace(headers={}, parse=lambda: "reply")
        completions.with_raw_response = SimpleNamespace(create=MagicMock(return_value=raw))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with observe_client_invocations(agent) as counter:
        assert _chat_completion_capturing_headers(agent, client, {}) == "reply"
        assert counter is not None and counter.snapshot()["count"] == 1
    if branch == "raw":
        create.assert_not_called()
        completions.with_raw_response.create.assert_called_once()
    else:
        create.assert_called_once()


@pytest.mark.parametrize("observation", [{"kind": "raw-private", "count": 8}, {"kind": {}, "count": 2},
    {"kind": "main_chat_client_invocation", "count": True}, None])
def test_invalid_observation_does_not_launder_a_legacy_iteration_count(observation):
    metadata = agent_end_metadata({"api_calls": 3, "api_call_observation": observation})
    assert (metadata["api_call_count"], metadata["api_call_count_kind"]) == (0, "unavailable")


def test_counter_is_bounded_and_metadata_prefers_it_without_changing_budget():
    with observe_client_invocations(_agent()) as counter:
        assert counter is not None
        counter.count = 1_000_000
        record_client_invocation()
        result = {"api_calls": 1, "api_call_observation": counter.snapshot()}
    metadata = agent_end_metadata(result)
    assert result["api_calls"] == 1
    assert (metadata["api_call_count"], metadata["api_call_count_kind"]) == (1_000_000, "main_chat_client_invocation")


def provider_failure(agent, monkeypatch, kind):
    agent._api_max_retries = 2
    agent._auto_recovery_cycles = 5
    error = (_Http(kind, f"HTTP {kind}: invalid api key" if kind in (401, 403) else f"HTTP {kind}: upstream failure")
             if isinstance(kind, int) else httpx.ConnectError("connect failed") if kind == "connect"
             else httpx.ReadTimeout("read timed out"))
    agent.client.chat.completions.create.side_effect = error
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0.0)
    with patch("agent.process_bootstrap.OpenAI", return_value=agent.client):
        result = agent.run_conversation("fixture")
    return result, agent.client.chat.completions.create.call_count


@pytest.mark.parametrize("kind,calls", [(401, 1), (403, 1), (429, 2), (500, 12), (503, 12),
                                       ("connect", 12), ("read", 12)])
def test_actual_retry_invocations_are_separate_from_the_iteration_budget(loop_agent, monkeypatch, kind, calls):
    result, observed = provider_failure(loop_agent, monkeypatch, kind)
    assert observed == calls
    assert result["api_calls"] == 1
    assert result["api_call_observation"] == {"kind": "main_chat_client_invocation", "count": calls}


def test_factory_failure_before_client_entry_is_observed_zero(loop_agent, monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("fixture-client-build-failure")

    monkeypatch.setattr(loop_agent, "_create_request_openai_client", unavailable)
    result = loop_agent.run_conversation("fixture")
    loop_agent.client.chat.completions.create.assert_not_called()
    assert result["api_call_observation"] == {"kind": "main_chat_client_invocation", "count": 0}


def partial_stream(agent, monkeypatch):
    def stream(**kwargs):
        yield SimpleNamespace(choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="visible fragment",
            tool_calls=None, reasoning_content=None, reasoning=None), finish_reason=None)], model="fixture", usage=None)
        raise httpx.ReadError("stream disconnected")

    agent.client.chat.completions.create.side_effect = stream
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: agent.client)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0.0)
    result = agent.run_conversation("fixture", stream_callback=lambda *args, **kwargs: None)
    return result, agent.client.chat.completions.create.call_count


def test_real_partial_stream_continuations_count_each_client_entry_once(loop_agent, monkeypatch):
    result, observed = partial_stream(loop_agent, monkeypatch)
    assert observed > 1
    assert result["api_call_observation"] == {"kind": "main_chat_client_invocation", "count": observed}
    assert result["partial"] is True
    assert "visible fragment" in result["final_response"]
    metadata = agent_end_metadata(result)
    assert (metadata["turn_exit_reason"], metadata["api_call_count"]) == ("api_incomplete_response", observed)


def test_reused_agent_starts_each_actual_turn_with_a_fresh_observation(loop_agent):
    before = loop_agent.client.chat.completions.create.call_count
    for _ in range(2):
        result = loop_agent.run_conversation("fixture")
        after = loop_agent.client.chat.completions.create.call_count
        assert after - before == 1
        assert result["api_call_observation"] == {"kind": "main_chat_client_invocation", "count": 1}
        assert result["api_calls"] == 1
        before = after


def test_uncaught_client_exit_resets_scope_and_never_labels_unknown_as_zero_actual(loop_agent, monkeypatch):
    from agent import observed_client_invocations as observation

    created = []
    original = observation.ClientInvocations

    def counter():
        value = original()
        created.append(value)
        return value

    monkeypatch.setattr(observation, "ClientInvocations", counter)
    loop_agent.platform = "cron"
    loop_agent.client.chat.completions.create.side_effect = GeneratorExit("fixture-client-exit")
    with pytest.raises(GeneratorExit):
        loop_agent.run_conversation("fixture")
    assert created[0].snapshot()["count"] == 1
    record_client_invocation()
    assert created[0].snapshot()["count"] == 1
    assert crash_metadata()["api_call_count_kind"] == "unavailable"
    loop_agent.client.chat.completions.create.side_effect = lambda **kwargs: _mock_response(content="ok")
    result = loop_agent.run_conversation("next fixture")
    assert result["api_call_observation"] == {"kind": "main_chat_client_invocation", "count": 1}
