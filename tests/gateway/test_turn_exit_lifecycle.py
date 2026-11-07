from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_agent_end_turn_exit_metadata import _runner


@pytest.mark.asyncio
async def test_stale_heartbeat_is_rejected_before_agent_start(monkeypatch):
    import gateway.run_heartbeat_acceptance as heartbeat

    runner, event, source, execute = _runner(monkeypatch, agent_result={"api_calls": 0})
    monkeypatch.setattr(heartbeat, "heartbeat_owner_is_current", lambda *args: False)
    assert await execute(event, source, "key", 1) is None
    runner.hooks.emit.assert_not_awaited()
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_end_hook_failure_cannot_deliver_an_error_for_superseded_turn(monkeypatch):
    runner, event, source, execute = _runner(monkeypatch, current=False,
        agent_result={"turn_exit_reason": "text_response(finish_reason=stop)", "api_calls": 2})
    emitted = AsyncMock()

    async def emit(name, context):
        await emitted(name, context)
        if name == "agent:end":
            raise RuntimeError("fixture-hook-failure")

    runner.hooks.emit = emit
    assert await execute(event, source, "key", 1) is None
    assert len([call for call in emitted.await_args_list if call.args[0] == "agent:end"]) == 1
    runner._hmwa_discard_stale_result.assert_called_once()
    runner._hmwa_agent_error_reply.assert_not_awaited()
