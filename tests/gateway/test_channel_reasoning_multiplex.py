import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tests.gateway.test_channel_reasoning_execution import _Constructed, _config, _runner, _source


def _multiplexer(tmp_path, monkeypatch, override=None):
    root = tmp_path / "root"
    operator = root / "profiles" / "gemma-operator"
    operator.mkdir(parents=True)
    configs: list[tuple[Path, dict[str, Any]]] = [(root, {}), (operator, _config())]
    for home, config in configs:
        config.update(model={"default": home.name}, agent={"reasoning_effort": "high"})
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    runner = _runner(override)
    del runner._load_reasoning_config
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_session_agent_runtime = lambda **kw: (
        gateway_run._load_gateway_config()["model"]["default"], {"api_key": "fixture-key"})
    return runner, root, operator


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,override", [("channel", None), ("thread", None), ("native-slash", None),
                                         ("thread", {"enabled": True, "effort": "low"})])
async def test_multiplex_foreground_reads_routed_config_at_construction(tmp_path, monkeypatch, kind, override):
    runner, root, operator = _multiplexer(tmp_path, monkeypatch, override)
    observed = []

    async def inner(message, context, history, source, session_id, **kwargs):
        turn = TurnRunner.__new__(TurnRunner)
        turn._runner = runner
        turn._ctx = TurnContext(source=source, session_key="fixture-session", message=message,
                                user_config=gateway_run._load_gateway_config(), enabled_toolsets=[])
        monkeypatch.setattr(turn, "_combined_ephemeral_prompt", lambda: "")
        monkeypatch.setattr(turn, "_setup_stream_consumer", lambda key: (None, None, None, False))

        def construct(route, platform, ephemeral, maximum, reasoning, routing):
            observed.append((route["model"], reasoning))
            raise _Constructed()

        monkeypatch.setattr(turn, "_resolve_turn_agent", construct)
        with pytest.raises(_Constructed):
            turn._run_sync_scoped()

    monkeypatch.setattr(runner, "_run_agent_inner", inner)
    for profile in ("default", "gemma-operator", "default"):
        source = _source(kind)
        source.profile = profile
        assert runner._resolve_profile_home_for_source(source) == (root if profile == "default" else operator)
        await runner._run_agent("complex", "", [], source, "fixture-session")
    manual = {"enabled": True, "effort": "low"}
    global_config = manual if override else {"enabled": True, "effort": "high"}
    channel_config = manual if override else {"enabled": True, "effort": "xhigh"}
    assert observed == [("root", global_config), ("gemma-operator", channel_config), ("root", global_config)]


@pytest.mark.asyncio
async def test_multiplex_background_reads_operator_policy_without_leaking_to_default(tmp_path, monkeypatch):
    import run_agent

    runner, _, _ = _multiplexer(tmp_path, monkeypatch)
    observed = []

    class Agent:
        def __init__(self, **kwargs):
            observed.append((kwargs["model"], kwargs["reasoning_config"]))

        def run_conversation(self, **kwargs):
            return {"final_response": "done", "messages": []}

    monkeypatch.setattr(run_agent, "AIAgent", Agent)
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda config: {})
    adapter = SimpleNamespace(send=AsyncMock(), extract_media=lambda text: ([], text), extract_images=lambda text: ([], text))
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *args: {})
    monkeypatch.setattr(runner, "_resolve_turn_toolsets", lambda *args: ([], None))
    monkeypatch.setattr(runner, "_refresh_fallback_model", lambda: None)
    monkeypatch.setattr(runner, "_cleanup_agent_resources", lambda agent: None)
    runner._session_db = None

    async def execute(callback):
        return callback()

    monkeypatch.setattr(runner, "_run_in_executor_with_context", execute)
    for profile in ("default", "gemma-operator", "default"):
        source = _source("thread")
        source.profile = profile
        await runner._run_background_task("complex", source, "fixture-task")
    assert observed == [("root", {"enabled": True, "effort": "high"}),
                        ("gemma-operator", {"enabled": True, "effort": "xhigh"}),
                        ("root", {"enabled": True, "effort": "high"})]
    assert adapter.send.await_count == 3
