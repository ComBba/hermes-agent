import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch, list_session_admissions


@pytest.mark.asyncio
async def test_initial_marker_is_latched_at_constructor_and_before_startup_recovery(tmp_path, monkeypatch):
    from gateway.run_startup import GatewayStartupMixin
    from hermes_constants import get_process_hermes_home
    marker = get_process_hermes_home()/'.drain_request.json'
    marker.write_text('{}')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    assert runner._external_drain_active is True
    assert runner._draining is False
    runner._external_drain_active = False
    seen = []
    class ReachedRecovery(Exception): pass
    runner._start_install_faulthandler = lambda: None
    async def log(): pass
    async def abort(adapter=None, platform=None): return False
    async def recover():
        seen.append(runner._external_drain_active)
        raise ReachedRecovery
    runner._start_log_startup_environment = log
    runner._abort_startup_if_shutdown_requested = abort
    runner._start_check_access_policy = lambda: False
    runner._start_recover_previous_run = recover
    with pytest.raises(ReachedRecovery):
        await GatewayStartupMixin._start_impl(runner)
    assert seen == [True]


@pytest.mark.asyncio
async def test_hold_rejects_new_submit_and_queued_claim_then_same_path_resumes(tmp_path):
    db = SessionDB(tmp_path/'state.db')
    db.create_session('s', source='telegram')
    ran = []
    async def handle(event):
        ran.append(event.text)
        return 'completed'
    runner = SimpleNamespace(_draining=False,_external_drain_active=True,_running=True,
        config=SimpleNamespace(multiplex_profiles=False),_adapter_for_source=lambda source:None,
        _handle_message=handle)
    authority = SessionAuthority(runner,profile_id='p',instance_id='owner',db=db,
                                 epoch=begin_runtime_epoch(db,instance_id='owner'))
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM,chat_id='c',user_id='human'),'route')
    actor = Principal('human','p',frozenset({'session:submit'}),'t')
    try:
        with pytest.raises(RuntimeStoreError,match='runtime_draining'):
            await authority.submit(actor,Submission('new',SessionRef('p','s'),{'text':'new'},'queue'),
                                   queue_while_restarting=True)
        admit_session_input(db,epoch=authority.epoch,principal_id='human',session_id='s',
                            request_id='queued',payload={'text':'queued'})
        authority._schedule(SessionRef('p','s'))
        if authority.sessions['s'].task:
            await asyncio.wait_for(authority.sessions['s'].task,5)
        assert [r['status'] for r in list_session_admissions(db,session_id='s')] == ['queued']
        assert ran == []
        runner._external_drain_active = False
        authority._schedule(SessionRef('p','s'))
        if authority.sessions['s'].task:
            await asyncio.wait_for(authority.sessions['s'].task,5)
        assert ran == ['queued']
    finally:
        db.close()


@pytest.mark.asyncio
async def test_hold_preserves_real_ws_attach_approval_clarify_and_refuses_new_writes(tmp_path):
    from gateway.session_controls import AuthorityConnection
    db = SessionDB(tmp_path/'state.db')
    db.create_session('s',source='test')
    runner = SimpleNamespace(_draining=False,_external_drain_active=True,_running=True)
    authority = SessionAuthority(runner,profile_id='p',instance_id='owner',db=db,
                                 epoch=begin_runtime_epoch(db,instance_id='owner'))
    authority.sessions['s'] = LiveSession(None,'route')
    viewer = AuthorityConnection(authority,object(),{'user_id':'human'},operator=True)
    async def call(method,**params):
        return await viewer.dispatch({'id':1,'method':method,'params':{'session_id':'s',**params}})
    try:
        assert 'result' in await call('session.resume')
        result = await call('approval.respond',execution_generation=1,prompt_id='gone',choice='once')
        assert result.get('error',{}).get('message') != 'runtime_draining'
        result = await viewer.dispatch({'id':2,'method':'clarify.lock','params':{
            'request_id':'missing','question_id':'q0','answer':'red'}})
        assert result['result']['status'] == 'expired'
        for method in ('prompt.submit','config.set','session.save','process.kill','reload.mcp'):
            assert (await call(method))['error']['message'] == 'runtime_draining'
    finally:
        await viewer.close()
        db.close()


@pytest.mark.asyncio
async def test_real_socket_ticket_and_receive_boundary_remains_open_for_existing_prompt(tmp_path,monkeypatch):
    import json
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_server as web,web_server_chat
    tickets = TicketStore('owner',frozenset({str(tmp_path)}))
    ticket = tickets.mint(profile_id=str(tmp_path),subject='human',purpose='interactive')
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path/'state.db'),profile_id=str(tmp_path),runner=None)
    runner = SimpleNamespace(_draining=False,_running=True,_external_drain_active=True,
        session_runtime_descriptor={'state':'ready'},session_authority=authority,session_ticket_store=tickets)
    received = []
    async def consume(ws,**kwargs):
        await ws.accept(subprotocol='hermes-gateway-v1')
        message = await ws.receive()
        received.append(message)
    async def fallback(*args):
        pytest.fail('native interactive ticket reached fallback')
    monkeypatch.setattr('tui_gateway.ws.handle_ws',consume)
    monkeypatch.setattr(web,'_DASHBOARD_EMBEDDED_CHAT_ENABLED',True)
    monkeypatch.setattr(web_server_chat,'_ws_request_is_allowed',lambda ws:True)
    frame = json.dumps({'method':'approval.respond','params':{'prompt_id':'parked'}})
    incoming = iter([{'type':'websocket.connect'},{'type':'websocket.receive','text':frame}])
    async def receive(): return next(incoming)
    sent = []
    async def send(value): sent.append(value)
    protocols = ['hermes-gateway-v1','hermes-gateway-ticket.'+ticket]
    scope = {'type':'websocket','path':'/api/ws','scheme':'ws','query_string':b'',
        'client':('127.0.0.1',1234),'server':('127.0.0.1',9999),
        'headers':[(b'sec-websocket-protocol',', '.join(protocols).encode())],'subprotocols':protocols}
    await GatewayRuntimeAPI(fallback,runner,web.app)(scope,receive,send)
    assert received == [{'type':'websocket.receive','text':frame}]
    assert sent[0]['type'] == 'websocket.accept'


@pytest.mark.asyncio
async def test_held_boot_and_reconnect_do_not_claim_or_send_then_release_reuses_recovery(tmp_path,monkeypatch):
    from unittest.mock import AsyncMock
    from gateway.run_startup import GatewayStartupMixin
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    runner._running = True
    runner._external_drain_active = True
    runner._claim_pending_obligations = AsyncMock(return_value=[])
    runner._send_restart_notification = AsyncMock()
    runner._replay_pending_planned_restart_notification = AsyncMock()
    monkeypatch.setattr('gateway.run._startup_restore_drain_timeout_secs',lambda:0)
    await runner._await_startup_boot_sends(planned_restart_notification_pending=True)
    assert await runner._redeliver_failed_obligations_for_platform(Platform.DISCORD) == 0
    runner._claim_pending_obligations.assert_not_called()
    runner._send_restart_notification.assert_not_called()
    runner._external_drain_active = False
    await runner._resume_external_hold_recovery()
    await runner._resume_external_hold_recovery()
    runner._claim_pending_obligations.assert_awaited_once()
    runner._send_restart_notification.assert_awaited_once()
    runner._replay_pending_planned_restart_notification.assert_awaited_once()


@pytest.mark.asyncio
async def test_held_startup_defers_actual_work_and_postconnect_dispatch_boundary(tmp_path):
    from unittest.mock import AsyncMock,Mock
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    runner._external_drain_active = True
    runner._start_register_plugins_relay_hooks = Mock()
    runner._start_relay_provision_retry = Mock()
    runner._subscribe_plugin_rewire = Mock()
    runner.hooks = SimpleNamespace(discover_and_load=Mock())
    runner._start_recover_previous_work = AsyncMock()
    runner._start_post_connect_services = AsyncMock()
    await runner._start_recover_previous_run()
    await runner._start_finish_wiring(0)
    runner._start_recover_previous_work.assert_not_called()
    runner._start_post_connect_services.assert_not_called()
    assert runner._managed_startup_recovery_held is True
    assert runner._managed_finish_wiring_held == 0
    assert runner._schedule_resume_pending_sessions() == 0


@pytest.mark.asyncio
async def test_concurrent_hold_release_resumes_deferred_startup_work_once(tmp_path):
    from unittest.mock import AsyncMock
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    runner._running = True
    runner._managed_startup_recovery_held = True
    async def restore(): await asyncio.sleep(0.01)
    runner._start_recover_previous_work = AsyncMock(side_effect=restore)
    await asyncio.gather(runner._resume_external_hold_recovery(),runner._resume_external_hold_recovery())
    runner._start_recover_previous_work.assert_awaited_once()


def test_external_hold_endpoint_is_distinct_and_keeps_session_authority(tmp_path):
    from hermes_cli.gateway_runtime import _endpoint
    home = str(tmp_path.resolve())
    payload = {'runtime_protocol':1,'instance_id':'owner','authority_epoch':1,'state':'draining',
        'drain_reason':'external_hold','api_origin':'http://127.0.0.1:41817','supervisor':'none',
        'served_profiles':[{'profile_id':home,'home':home}],'capabilities':['session-authority-v1']}
    held = _endpoint(payload,tmp_path)
    assert (held.state,held.reason_code) == ('draining','external_hold')
    assert held.endpoint is not None
    payload['capabilities'] = []
    assert _endpoint(payload,tmp_path).endpoint is None


@pytest.mark.asyncio
async def test_hold_arriving_after_ledger_claim_releases_without_spending_attempt_and_retries_once(tmp_path,monkeypatch):
    from gateway import delivery_ledger as ledger
    from unittest.mock import AsyncMock
    monkeypatch.setattr(ledger,'_db_path',lambda:tmp_path/'ledger.db')
    ledger.record_obligation(obligation_id='owed',session_key='',platform='discord',chat_id='42',
                             thread_id=None,content='original completed text')
    ledger.mark_attempting('owed')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    send = AsyncMock(return_value=SimpleNamespace(success=True,message_id='actual-ack'))
    runner.adapters = {Platform.DISCORD:SimpleNamespace(send=send)}
    runner._running = True
    runner._external_drain_active = True
    row = {'obligation_id':'owed','session_key':'','platform':'discord','chat_id':'42','thread_id':None,
           'content':'original completed text','attempts':1,'profile':None,'needs_marker':False}
    assert await runner._redeliver_claimed_obligations([row]) == 0
    send.assert_not_called()
    with ledger._connect() as connection:
        saved = connection.execute('SELECT state,attempts FROM delivery_obligations WHERE obligation_id="owed"').fetchone()
    assert tuple(saved) == ('failed',0)
    runner._external_drain_active = False
    await runner._resume_external_hold_recovery()
    await runner._resume_external_hold_recovery()
    assert send.await_count == 1
    with ledger._connect() as connection:
        assert connection.execute('SELECT state FROM delivery_obligations WHERE obligation_id="owed"').fetchone()[0] == 'delivered'
