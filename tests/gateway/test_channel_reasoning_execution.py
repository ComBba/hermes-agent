from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run as gateway_run, session_policy
from gateway.channel_reasoning import resolve_channel_reasoning_config
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.run_background import GatewayBackgroundMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class _Constructed(Exception):
    pass


def _config():
    return {"discord": {"channel_reasoning_policies": {"private-parent": {
        "default_effort": "medium", "xhigh_terms": ["complex"],
        "ultra": {"enabled": True, "terms": ["parallel"], "requires_thread": True, "blocked_terms": ["deploy"]},
    }}}}


def _source(kind):
    threaded = kind != "channel"
    return SessionSource(
        platform=Platform.DISCORD, chat_id="private-thread" if threaded else "private-parent",
        thread_id="private-thread" if threaded else None, parent_chat_id="private-parent" if threaded else None,
        user_id="fixture-actor", user_id_alt=None, user_name="fixture", chat_name="fixture", chat_type="thread",
    )


def _runner(override=None, model="model-a"):
    state = SimpleNamespace(conversation=SimpleNamespace(reasoning_override=override)) if override is not None else None
    runner = GatewayRunner.__new__(GatewayRunner)
    attributes = {
        "_peek_session_state": lambda key: state,
        "_resolve_session_key_or_none": lambda source, session_key: session_key,
        "_load_reasoning_config": lambda model: {"enabled": True, "effort": "high"},
        "_provider_routing": {}, "_resolve_session_agent_runtime": lambda **kw: (model, {"api_key": "fixture-key"}),
        "_resolve_session_service_tier": lambda **kw: None,
        "_resolve_turn_agent_config": lambda message, chosen, runtime: {"model": chosen, "runtime": runtime},
    }
    for name, value in attributes.items():
        setattr(runner, name, value)
    return runner


@pytest.mark.parametrize("kind,message,override,expected", [
    ("channel", "ordinary", None, "medium"), ("thread", "complex", None, "xhigh"),
    ("forum", "parallel", None, "ultra"), ("native-slash", "parallel deploy", None, "xhigh"),
    ("thread", "complex", {"enabled": True, "effort": "low"}, "low"),
])
def test_foreground_turn_passes_effective_policy_to_actual_agent_construction(monkeypatch, caplog, kind, message, override, expected):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", _config)
    monkeypatch.setattr(session_policy, "policy_for_source", lambda *a: None)
    runner = _runner(override, model="effective-model-after-switch")
    turn = TurnRunner.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = TurnContext(source=_source(kind), session_key="fixture-session", user_config={}, message=message,
                                inbound_message_id=None, enabled_toolsets=[])
    monkeypatch.setattr(turn, "_combined_ephemeral_prompt", lambda: "")
    monkeypatch.setattr(turn, "_setup_stream_consumer", lambda platform_key: (None, None, None, False))
    observed = {}

    def construct(route, platform, ephemeral, maximum, reasoning, routing):
        observed.update(model=route["model"], reasoning=reasoning)
        raise _Constructed()

    monkeypatch.setattr(turn, "_resolve_turn_agent", construct)
    with caplog.at_level("INFO"), pytest.raises(_Constructed):
        turn._run_sync_scoped()
    assert observed == {"model": "effective-model-after-switch", "reasoning": {"enabled": True, "effort": expected}}
    assert "private-parent" not in caplog.text and "private-thread" not in caplog.text
    if override is not None:
        assert "channel reasoning resolved" not in caplog.text


@pytest.mark.asyncio
async def test_background_turn_constructs_agent_with_channel_policy_and_keeps_model(monkeypatch, caplog):
    import run_agent

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _config)
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda config: {})
    monkeypatch.setattr(session_policy, "policy_for_source", lambda *a: None)
    observed = {}

    class Agent:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def run_conversation(self, **kwargs):
            return {"final_response": "done", "messages": []}

    monkeypatch.setattr(run_agent, "AIAgent", Agent)
    adapter = SimpleNamespace(send=AsyncMock(), extract_media=lambda text: ([], text), extract_images=lambda text: ([], text))
    runner = _runner()
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *args: {})
    monkeypatch.setattr(runner, "_resolve_turn_toolsets", lambda *args: ([], None))
    monkeypatch.setattr(runner, "_refresh_fallback_model", lambda: None)
    monkeypatch.setattr(runner, "_cleanup_agent_resources", lambda agent: None)
    runner._session_db = None

    async def execute(callback):
        return callback()

    monkeypatch.setattr(runner, "_run_in_executor_with_context", execute)
    with caplog.at_level("INFO"):
        await GatewayBackgroundMixin._run_background_task_inner(runner, "complex", _source("thread"), "fixture-task")
    assert observed["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    assert observed["model"] == "model-a"
    assert adapter.send.await_count == 1
    assert "private-parent" not in caplog.text and "private-thread" not in caplog.text


@pytest.mark.parametrize("config", [None, [], {"discord": True}, {"discord": {"channel_reasoning_policies": []}},
    {"discord": {"channel_reasoning_policies": {"private-parent": {"default_effort": True}}}}])
def test_invalid_config_shape_does_not_create_a_policy(config):
    assert resolve_channel_reasoning_config(config, platform="discord", chat_id="private-parent") is None


def test_non_string_terms_cannot_escalate_a_bound_channel():
    config = {"discord": {"channel_reasoning_policies": {"private-parent": {
        "default_effort": "medium", "xhigh_terms": [42, False, {"term": "ordinary"}],
    }}}}
    result = resolve_channel_reasoning_config(config, platform="discord", chat_id="private-parent", message="42 false ordinary")
    assert result == {"enabled": True, "effort": "medium"}
