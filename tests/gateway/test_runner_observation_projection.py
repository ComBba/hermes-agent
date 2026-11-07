from datetime import UTC, datetime, timedelta
import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from agent.observed_client_invocations import ClientInvocations
from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from gateway.turn_exit_metadata import agent_end_metadata, validated_api_call_observation
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from hermes_state_terminal import RESULT_PREFIX, load_result
from plugins.platforms.discord.adapter import DiscordAdapter
from run_agent import AIAgent


@pytest.mark.parametrize("kind", ["main_chat_client_invocation", "codex_turn_attempt"])
@pytest.mark.parametrize("count", [0, 1_000_000])
def test_valid_observation_is_a_detached_closed_value(kind, count):
    value = {"kind": kind, "count": count}
    observed = validated_api_call_observation(value)
    assert observed == value and observed is not value


@pytest.mark.parametrize("value", [None, [], {"kind": "unknown", "count": 1},
    {"kind": "main_chat_client_invocation", "count": True},
    {"kind": "main_chat_client_invocation", "count": -1},
    {"kind": "main_chat_client_invocation", "count": 1_000_001},
    {"kind": "main_chat_client_invocation", "count": "1"},
    {"kind": "main_chat_client_invocation", "count": 1.0},
    {"kind": "main_chat_client_invocation", "count": object()},
    {"kind": "main_chat_client_invocation", "count": 1, "PRIVATE": "fixture-token"}])
def test_invalid_observation_is_rejected_without_legacy_laundering(value):
    assert validated_api_call_observation(value) is None
    metadata = agent_end_metadata({"api_calls": 7, "api_call_observation": value})
    assert (metadata["api_call_count"], metadata["api_call_count_kind"]) == (0, "unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize("observation", ["actual",
    {"kind": "main_chat_client_invocation", "count": True},
    {"kind": "main_chat_client_invocation", "count": 1_000_001},
    {"kind": "main_chat_client_invocation", "count": object()},
    {"kind": "main_chat_client_invocation", "count": 1, "PRIVATE": "fixture-token"}],
    ids=["actual-sdk401", "bool-count", "oversize-count", "non-json-count", "private-field"])
async def test_sdk_failure_survives_real_turnrunner_native_persistence_and_consumers(monkeypatch, tmp_path, observation):
    home = get_hermes_home()
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setenv("HERMES_TURN_EXIT_DIR", str(tmp_path / "turn-exits"))
    (home / "config.yaml").write_text(json.dumps({"model": {"default": "gpt-4.1", "provider": "openai",
        "base_url": "https://model.invalid/v1", "context_length": 131072},
        "managed_inquiry": {"enabled": False}, "display": {"streaming": False}}))
    (home / ".env").write_text("OPENAI_API_KEY=fixture-key\nOPENAI_BASE_URL=https://model.invalid/v1\n")
    carrier = Path(gateway_run.__file__).resolve().parents[1] / "managed_release_payload"
    hook = home / "hooks/turn-exit-monitor"
    hook.mkdir(parents=True, exist_ok=True)
    for name in ("HOOK.yaml", "handler.py"):
        (hook / name).write_bytes((carrier / "hooks/turn-exit-monitor" / name).read_bytes())
    requests = []
    clients = []

    def http_client(*args, **kwargs):
        def response(request):
            requests.append(request.url.path)
            return httpx.Response(401, json={"error": {"message": "invalid api key", "code": "invalid_api_key"}})
        client = httpx.Client(transport=httpx.MockTransport(response))
        clients.append(client)
        return client

    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", http_client)
    if observation != "actual":
        monkeypatch.setattr(ClientInvocations, "snapshot", lambda self: observation)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = gateway_run.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture", typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    monkeypatch.setattr(adapter, "gateway_runner", runner)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="projection-regression")
    runner._wire_adapter_handlers(adapter)
    runner.hooks.discover_and_load()
    emitted = []
    original_emit = runner.hooks.emit

    async def emit(name, context=None):
        if name == "agent:end":
            emitted.append(dict(context or {}))
        return await original_emit(name, context)

    monkeypatch.setattr(runner.hooks, "emit", emit)
    monkeypatch.setattr(adapter, "send", AsyncMock(return_value=SendResult(success=True, message_id="fixture-ack")))
    try:
        event = MessageEvent("fixture", source=adapter.build_source(chat_id="42", chat_type="dm", user_id="42"),
                             message_id="projection-message", timestamp=datetime.now(UTC))
        with native_callback(runner, event, home):
            receipt = await authority.admit_native(event)
        await authority._drain(receipt.ref)
        admission = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)[0]
        with authority.db._read_ctx() as db:
            row = db.execute("SELECT value FROM state_meta WHERE key=?", (RESULT_PREFIX + receipt.admission_id,)).fetchone()
        assert admission["status"] == "terminal" and row is not None
        result: dict[str, Any] = load_result(row["value"])["result"]
        assert requests == ["/v1/chat/completions"]
        assert result["api_calls"] == 1
        expected = {"kind": "main_chat_client_invocation", "count": 1} if observation == "actual" else None
        assert result["api_call_observation"] == expected
        kind, count = ("main_chat_client_invocation", 1) if observation == "actual" else ("unavailable", 0)
        assert len(emitted) == 1
        assert (emitted[0]["api_call_count_kind"], emitted[0]["api_call_count"]) == (kind, count)
        ledger = tmp_path / "turn-exits/events.jsonl"
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["api_call_count_kind"] == kind
        spec = importlib.util.spec_from_file_location("projection_collector", carrier / "scripts/collect_turn_exits.py")
        assert spec is not None and spec.loader is not None
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        alerts = collector.collect_turn_exit_alerts(ledger, observed_now=datetime.now().astimezone() + timedelta(seconds=1))
        assert len(alerts) == 1 and f"count_kind={kind}" in alerts[0]["evidence"]
        assert "PRIVATE" not in json.dumps(result) + ledger.read_text() + json.dumps(alerts)
        assert "fixture-token" not in json.dumps(result) + ledger.read_text() + json.dumps(alerts)
    finally:
        for client in clients:
            client.close()
        authority.db.close()
