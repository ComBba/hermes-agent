import asyncio
import json
import inspect
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import pytest


async def main():
    from tests.gateway.test_managed_inquiry_bridge import external_io,owner,run_native_inquiry
    from gateway import managed_inquiry_bridge as bridge
    from gateway.managed_ingress_provenance import AdmissionOwner,PlatformObservation,bind_admission,execution_admission,verified_ingress
    from gateway.managed_inquiry.task_models import BindingRole,InquiryRequest,TaskScope
    from gateway.managed_inquiry.task_state import PendingStore
    from gateway.platforms.event import MessageEvent
    from gateway.session_envelope import restore_native
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input
    from datetime import UTC,datetime
    patch = pytest.MonkeyPatch()
    external = inspect.unwrap(external_io)(patch)
    io = next(external)
    home = get_hermes_home()
    temporary = home/'fixture'
    temporary.mkdir(parents=True,exist_ok=True)
    fixture = inspect.unwrap(owner)(temporary,patch,io)
    runtime = await fixture.__anext__()
    runner,adapter,authority = runtime
    try:
        if sys.argv[1] in {'control','cold'}:
            from gateway.managed_loaded_receipt import seal_loaded_receipt
            from gateway.control_socket import query_gateway_control
            from gateway.run_bootstrap import _start_gateway_start_control_socket
            from gateway.runtime_bootstrap import TicketStore
            (home/'config.yaml').write_text(json.dumps({'managed_inquiry':{'schema_version':1,
                'enabled':True,'profile_id':authority.profile_id,'platform':'discord',
                'inquiry_channels':['42'],'coordinator_channels':[]}}))
            runner._external_drain_active = True
            await runner._start_finish_wiring(0)
            runner._running = True
            runner.session_ticket_store = TicketStore('owner',frozenset({authority.profile_id}))
            runner.session_runtime_descriptor = {'runtime_protocol':1,'instance_id':'owner',
                'authority_epoch':authority.epoch,'state':'ready','api_origin':'http://127.0.0.1:41817',
                'served_profiles':[{'profile_id':authority.profile_id,'home':str(home)}],
                'capabilities':['session-authority-v1']}
            seal_loaded_receipt(runner)
            server = await _start_gateway_start_control_socket(runner)
            assert server is not None
            path = Path(bridge.__file__)
            original = path.read_bytes()
            try:
                before = await asyncio.to_thread(query_gateway_control,home,'managed-loaded-receipt')
                if sys.argv[1]=='cold':
                    cold={'gets':len(io.gets),'sends':len(io.sends),'discoveries':io.discoveries,
                          'pending_db':(home/'managed-inquiry/state.sqlite').exists()}
                    runner._external_drain_active = False
                    terminal=await run_native_inquiry(runtime,patch)
                    assert terminal[-1]['managed_inquiry_v1']['store_verified'] is True
                    after=await asyncio.to_thread(query_gateway_control,home,'managed-loaded-receipt')
                else:
                    cold=None
                    path.write_bytes(original.replace(b"'ingress_blocked'",b"'ingress_changed'",1))
                    after=await asyncio.to_thread(query_gateway_control,home,'managed-loaded-receipt')
                summary = {'pid':os.getpid(),'before':before,'after':after,
                    'cold':cold,'sends':len(io.sends),'gets':len(io.gets),'discoveries':io.discoveries}
            finally:
                path.write_bytes(original)
                await server.stop()
        elif sys.argv[1]=='produce':
            io.send_success = False
            _,_,_,_,receipt,row,result = await run_native_inquiry(runtime,patch)
            assert result['managed_inquiry_v1']['store_verified'] is True
            assert row['status']=='terminal'
            key = bridge.managed_completion_key(adapter._native_test_events[-1],result['final_response'])
            assert key is not None
            summary = {'pid':os.getpid(),'sends':len(io.sends),'gets':len(io.gets),
                'historical_epoch':row['owner_epoch'],'admission_id':receipt.admission_id,
                'scope':json.loads(key)['scope']}
            (home/'produce.json').write_text(json.dumps(summary))
        else:
            previous = json.loads((home/'produce.json').read_text())
            io.send_success = True
            observed = []
            actual = bridge.record_replayed_delivery
            def observe(*args):
                value = actual(*args)
                observed.append(value)
                return value
            patch.setattr(bridge,'record_replayed_delivery',observe)
            claimed = await runner._claim_pending_obligations()
            delivered = await runner._redeliver_claimed_obligations(claimed)
            scope = TaskScope(**previous['scope'])
            source = adapter.build_source(chat_id=scope.channel_id,chat_type='dm',user_id=scope.actor_id,
                                          thread_id=scope.thread_id or None)
            event = MessageEvent('상태 확인',source=source,message_id='reply-after-restart',
                timestamp=datetime.now(UTC),reply_to_message_id='ack-1')
            event._managed_platform_observation = PlatformObservation(True,'reply-after-restart',scope.actor_id,(),'ack-1')
            with native_callback(runner,event,home):
                reply = await authority.admit_native(event)
            row = claim_session_input(authority.db,epoch=authority.epoch,session_id=reply.ref.session_id)
            assert isinstance(row,dict)
            restored = restore_native(row['payload'],runner)
            bind_admission(restored,row,AdmissionOwner(authority.profile_id,reply.ref.session_id,authority.epoch))
            restored.__dict__['_claimed_owner'] = (reply.ref.session_id,row['generation'])
            ingress = verified_ingress(restored,session_id=reply.ref.session_id).provenance
            execution = execution_admission(restored,session_id=reply.ref.session_id)
            request = InquiryRequest(scope,BindingRole.INQUIRY,restored.text,'reply-after-restart',
                execution,ingress,reply_to_message_id='ack-1')
            reference = PendingStore(home/'managed-inquiry/state.sqlite').lookup_reply(request)
            summary = {'pid':os.getpid(),'sends':len(io.sends),'gets':len(io.gets),'claimed':len(claimed),
                'delivered':delivered,'replay_verified':observed,'reply_linked':reference is not None,
                'historical_epoch':previous['historical_epoch'],'current_epoch':authority.epoch}
        print('NATIVE_RECEIPT='+json.dumps(summary,sort_keys=True))
    finally:
        await fixture.aclose()
        external.close()
        patch.undo()


if __name__=='__main__':
    asyncio.run(main())
