from copy import deepcopy
import json
import sqlite3
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.managed_ingress_provenance import AdmissionOwner, bind_admission, execution_admission
from gateway.managed_inquiry_bridge import _capture, verified_managed_completion
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.run_startup import GatewayStartupMixin
from tests.gateway.test_delivery_ledger_producer import _Adapter, _event


def _completion():
    event = _event("/inquiry advice")
    event.source.profile = "default"
    event.source.user_id = "fixture-actor"
    principal = "messaging:" + json.dumps([event.source.profile, event.source.platform.value,
        event.source.chat_id, event.source.thread_id, event.source.user_id], separators=(",", ":"))
    bind_admission(event, {"payload": {"native_text_v1": {}}, "status": "started", "owner_epoch": 2,
        "generation": 3, "session_id": "session", "admission_id": "admission", "principal_id": principal,
        "request_id": "request"}, AdmissionOwner("default", "session", 2))
    event.__dict__["_claimed_owner"] = ("session", 3)
    execution = execution_admission(event, session_id="session")
    text = _capture(event, execution, "fixture managed reply",
                    {"completion_id": "fixture-completion", "receipt_sha256": "fixture-receipt"})
    return event, text


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (12345, 1.0))
    return tmp_path / "state.db"


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [None, "text", "actor", "generation", "message", "forged"])
async def test_only_owner_bound_managed_slash_completion_is_ledgered(ledger, tamper):
    event, text = _completion()
    if tamper == "text":
        text += " changed"
    elif tamper == "actor":
        event.source.user_id = "other"
    elif tamper == "generation":
        event.__dict__["_claimed_owner"] = ("session", 4)
    elif tamper == "message":
        event.message_id = "other"
    elif tamper == "forged":
        event.__dict__["_managed_completion_capability"] = {"store_verified": True}
    adapter = _Adapter()
    obligation = await adapter._record_delivery_obligation(event, "key", text, adapter, False)
    assert (obligation is not None) == (tamper is None)
    assert verified_managed_completion(event, text) == (tamper is None)


@pytest.mark.asyncio
async def test_failed_managed_send_preserves_real_ledger_and_success_finalizes_it(ledger, monkeypatch):
    event, text = _completion()
    adapter = _Adapter()
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=False, error="send_path_degraded"))
    result, _ = await adapter.send_final_ledgered(event, "key", text, {}, reply_to=None)
    assert result.success is False
    obligation_id = dl.compute_obligation_id("key", str(event.message_id), text)
    with dl._connect() as db:
        row = db.execute("SELECT state, attempts FROM delivery_obligations WHERE obligation_id=?", (obligation_id,)).fetchone()
    assert row == ("failed", 0)
    claims = dl.sweep_failed_for_runtime("slack")
    assert len(claims) == 1 and claims[0]["attempts"] == 1
    assert dl.sweep_failed_for_runtime("slack") == []
    runner = GatewayRunner.__new__(GatewayRunner)
    monkeypatch.setattr(runner, "_obligation_adapter", AsyncMock(return_value=adapter))
    monkeypatch.setattr(runner, "_arm_flood_timers_for_waiting_rows", AsyncMock())
    assert await GatewayStartupMixin._redeliver_claimed_obligations(runner, claims) == 1
    with dl._connect() as db:
        rows = db.execute("SELECT state FROM delivery_obligations").fetchall()
    assert rows == [("delivered",)]
    assert dl.sweep_failed_for_runtime("slack") == []
    assert len(adapter.sent) == 1
    assert adapter._send_with_retry.await_count == 1


@pytest.mark.asyncio
async def test_unmanaged_and_ephemeral_slash_keep_existing_skip(ledger):
    event, text = _completion()
    adapter = _Adapter()
    assert await adapter._record_delivery_obligation(event, "key", text, adapter, True) is None
    unmanaged = deepcopy(event)
    delattr(unmanaged, "_managed_completion_capability")
    assert await adapter._record_delivery_obligation(unmanaged, "key", text, adapter, False) is None


@pytest.mark.parametrize("key", [None, "opaque-managed-completion-key"])
@pytest.mark.parametrize("recovery", ["runtime", "restart"])
def test_existing_ledger_preserves_optional_completion_key_across_claim(ledger, monkeypatch, key, recovery):
    dl.record_obligation(obligation_id="owed", session_key="session", platform="slack", chat_id="chat",
                         thread_id=None, content="reply", managed_completion_key=key)
    dl.mark_failed("owed", "send_path_degraded")
    if recovery == "restart":
        monkeypatch.setattr(dl, "_owner_alive", lambda *args: False)
        rows = dl.sweep_recoverable()
    else:
        rows = dl.sweep_failed_for_runtime("slack")
    assert len(rows) == 1
    assert rows[0].get("managed_completion_key") == key
    assert rows[0]["attempts"] == 1


def test_preexisting_ledger_schema_migrates_with_no_completion_authority(ledger):
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE delivery_obligations (obligation_id TEXT PRIMARY KEY, session_key TEXT, "
                   "platform TEXT, chat_id TEXT, thread_id TEXT, content TEXT, state TEXT, attempts INTEGER, "
                   "created_at REAL, updated_at REAL, owner_pid INTEGER, owner_started_at INTEGER, last_error TEXT, "
                   "adapter_profile TEXT)")
        db.execute("INSERT INTO delivery_obligations VALUES ('old','s','slack','c',NULL,'reply','failed',0,0,0,NULL,NULL,NULL,'default')")
    with dl._connect() as db:
        assert db.execute("SELECT managed_completion_key FROM delivery_obligations").fetchone() == (None,)
