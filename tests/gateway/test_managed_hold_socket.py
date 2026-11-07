import asyncio
from datetime import UTC,datetime
import json
import threading

import pytest

from tests.gateway import test_managed_inquiry_bridge as native

owner = native.owner
external_io = native.external_io
run_native_inquiry = native.run_native_inquiry


@pytest.mark.asyncio
async def test_real_ticket_socket_answers_parked_approval_clarify_and_completes_under_hold(owner,monkeypatch,external_io):
    from gateway.managed_ingress_provenance import PlatformObservation,AdmissionOwner,bind_admission
    from gateway.platforms.event import MessageEvent
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from gateway.session_envelope import restore_native
    from gateway.session_ingress_context import native_callback
    from gateway.session_ingress import deliver_response
    from gateway.session_results import finish_result
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input,list_session_admissions
    from hermes_cli import web_server as web
    from tools.approval_gateway_wait import _await_gateway_decision
    from tools import approval,clarify_gateway
    runner,adapter,authority,_,first,_,_ = await run_native_inquiry(owner,monkeypatch)
    sid = first.ref.session_id
    route = authority.sessions[sid].route
    async def admit(message_id,text):
        event = MessageEvent(text,source=adapter.build_source(chat_id='42',chat_type='dm',user_id='42'),
            message_id=message_id,timestamp=datetime.now(UTC))
        event._managed_platform_observation = PlatformObservation(True,message_id,'42',())
        with native_callback(runner,event,get_hermes_home()):
            return await authority.admit_native(event)
    await admit('parked','승인과 추가 답변을 기다리는 기존 작업')
    row = claim_session_input(authority.db,epoch=authority.epoch,session_id=sid)
    assert isinstance(row,dict)
    future = await admit('queued-before-hold','상태 재확인')
    runner._running = True
    runner._external_drain_active = True
    runner.session_runtime_descriptor = {'state':'ready'}
    tickets = runner.session_ticket_store = TicketStore('owner',frozenset({authority.profile_id}))
    ticket = tickets.mint(profile_id=authority.profile_id,subject='human',purpose='interactive')
    monkeypatch.setattr(web,'_DASHBOARD_EMBEDDED_CHAT_ENABLED',True)
    inbox = asyncio.Queue()
    responses = {}
    accepted = asyncio.Event()
    async def send(frame):
        if frame['type']=='websocket.accept': accepted.set()
        for line in (frame.get('text') or '').splitlines():
            decoded = json.loads(line)
            if 'id' in decoded: responses[decoded['id']] = decoded
    async def request(rid,method,**params):
        await inbox.put({'type':'websocket.receive','text':json.dumps({'jsonrpc':'2.0','id':rid,
            'method':method,'params':{'session_id':sid,**params}})})
        deadline = asyncio.get_running_loop().time()+3
        while rid not in responses and asyncio.get_running_loop().time()<deadline:
            await asyncio.sleep(.005)
        assert rid in responses,(rid,responses)
        return responses[rid]
    async def fallback(*args): pytest.fail('ticketed native socket reached fallback')
    protocols = ['hermes-gateway-v1','hermes-gateway-ticket.'+ticket]
    scope = {'type':'websocket','path':'/api/ws','scheme':'ws','query_string':b'',
        'client':('127.0.0.1',1234),'server':('127.0.0.1',9999),
        'headers':[(b'host',b'127.0.0.1:9999'),(b'sec-websocket-protocol',', '.join(protocols).encode())],
        'subprotocols':protocols}
    await inbox.put({'type':'websocket.connect'})
    socket = asyncio.create_task(GatewayRuntimeAPI(fallback,runner,web.app)(scope,inbox.get,send))
    approval_wait = None
    clarify_wait = None
    try:
        await asyncio.wait_for(accepted.wait(),3)
        assert 'result' in await request(1,'session.resume')
        assert (await request(2,'prompt.submit',input_id='held-new',text='new'))['error']['message']=='runtime_draining'
        published = threading.Event()
        def notify(data):
            authority.register_approval(sid,row['generation'],route,data)
            published.set()
        approval_wait = asyncio.create_task(asyncio.to_thread(_await_gateway_decision,route,notify,
            {'request_id':'parked-approval','command':'fixture effect awaiting approval'}))
        assert await asyncio.to_thread(published.wait,2)
        approved = await request(3,'approval.respond',execution_generation=row['generation'],
                                 prompt_id='parked-approval',choice='once')
        assert approved['result']['status']=='resolved'
        assert (await asyncio.wait_for(approval_wait,2))['choice']=='once'
        entry = clarify_gateway.register('parked-clarify',route,'선택해주세요',['blue','red'])
        authority.register_clarify(sid,row['generation'],entry)
        clarify_wait = asyncio.create_task(asyncio.to_thread(clarify_gateway.wait_for_response,'parked-clarify',2))
        clarified = await request(4,'clarify.respond',execution_generation=row['generation'],
                                  prompt_id='parked-clarify',answer='blue')
        assert clarified['result']['status']=='resolved'
        assert await asyncio.wait_for(clarify_wait,2)=='blue'
        completed,_ = finish_result(authority.db,epoch=authority.epoch,row=row,response='기존 작업 완료',
            outcome='completed',result={'result':{'final_response':'기존 작업 완료','messages':[],'completed':True}})
        assert completed['status']=='terminal'
        restored = restore_native(row['payload'],runner)
        bind_admission(restored,row,AdmissionOwner(authority.profile_id,sid,authority.epoch))
        restored.__dict__['_claimed_owner'] = (sid,row['generation'])
        assert await deliver_response(adapter,restored,route,'기존 작업 완료') is True
        await authority._drain(first.ref)
        queued = next(item for item in list_session_admissions(authority.db,session_id=sid)
                      if item['admission_id']==future.admission_id)
        assert queued['status']=='queued'
        runner._external_drain_active = False
        await authority._drain(first.ref)
        after = next(item for item in list_session_admissions(authority.db,session_id=sid,pending_only=False)
                     if item['admission_id']==future.admission_id)
        assert after['status']=='terminal' and after['outcome']=='completed'
        runner._hmwa_prepare_turn.assert_not_called()
    finally:
        approval.resolve_gateway_approval(route,'deny',request_id='parked-approval')
        clarify_gateway.resolve_gateway_clarify('parked-clarify','red')
        await inbox.put({'type':'websocket.disconnect','code':1000})
        await asyncio.wait_for(socket,3)
        if approval_wait is not None:
            await asyncio.wait_for(approval_wait,3)
        if clarify_wait is not None:
            await asyncio.wait_for(clarify_wait,3)
