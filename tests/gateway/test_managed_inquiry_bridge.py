import json
import asyncio
import io
import threading
import time
import hashlib
import sqlite3
from dataclasses import replace
from datetime import UTC,datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.managed_ingress_provenance import PlatformObservation
from gateway.platforms.event import MessageEvent
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from hermes_state_terminal import RESULT_PREFIX,load_result


@pytest.fixture(autouse=True)
def external_io(monkeypatch):
    from gateway.managed_inquiry.discovery import MacDiscovery
    from gateway.managed_inquiry.models import Discovery,ServiceIdentity,Sha256,TargetEvidence
    state = SimpleNamespace(discoveries=0,gets=[],sends=[],send_success=True,
        body=b'{"ok":true,"service":"fixture","data_age_sec":0}',status=200,
        hold_close=False,closed=threading.Event(),release=threading.Event())
    def discover(self):
        state.discoveries += 1
        return Discovery((TargetEvidence('http://127.0.0.1:41817',Sha256('a'*64),time.time(),
                                         ServiceIdentity('service','fixture')),))
    class Response:
        def __init__(self):
            self.code = state.status
            self.headers = {'Content-Type':'application/json'}
            self.stream = io.BytesIO(state.body)
        def read1(self,size): return self.stream.read(size)
        def close(self):
            state.closed.set()
            if state.hold_close:
                assert state.release.wait(3),'external transport facade was not released'
    class Opener:
        def open(self,request,timeout):
            assert request.full_url == 'http://127.0.0.1:41817/healthz'
            assert request.get_method() == 'GET'
            assert 0 < timeout <= 15
            state.gets.append(request.full_url)
            return Response()
    monkeypatch.setattr(MacDiscovery,'__call__',discover)
    monkeypatch.setattr('gateway.managed_inquiry.transport.request.build_opener',lambda *args:Opener())
    yield state
    state.release.set()


@pytest_asyncio.fixture
async def owner(tmp_path,monkeypatch,external_io):
    from gateway.config import GatewayConfig,Platform,PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority,initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setenv('DISCORD_ALLOWED_USERS','42,43')
    monkeypatch.setattr(SessionAuthority,'_schedule',lambda self,ref:None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    adapter = DiscordAdapter(PlatformConfig(enabled=True,token='fixture',typing_indicator=False))
    monkeypatch.setattr(adapter,'gateway_runner',runner)
    runner.adapters = {Platform.DISCORD:adapter}
    authority = await initialize_session_authority(runner,profile_id=str(get_hermes_home()),instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    events = []
    monkeypatch.setattr(adapter,'_native_test_events',events,raising=False)
    external_io.finalized = asyncio.Event()
    original_final = adapter.send_final_ledgered
    async def observe_final(*args,**kwargs):
        events.append(args[0])
        result = await original_final(*args,**kwargs)
        external_io.finalized.set()
        return result
    monkeypatch.setattr(adapter,'send_final_ledgered',observe_final)
    async def send(chat_id,content,reply_to=None,metadata=None):
        external_io.sends.append((chat_id,content,metadata))
        return SendResult(success=external_io.send_success,message_id='ack-1' if external_io.send_success else None,
                          error=None if external_io.send_success else 'fixture_offline')
    adapter.send = AsyncMock(side_effect=send)
    runner._hmwa_prepare_turn = AsyncMock(side_effect=AssertionError('managed handler fell through to agent'))
    try:
        yield runner,adapter,authority
    finally:
        external_io.release.set()
        authority.db.close()


def configure_binding(authority,inquiry=('42',),coordinator=(),extras=None):
    (get_hermes_home()/'config.yaml').write_text(json.dumps({'managed_inquiry':{'schema_version':1,'enabled':True,
        'profile_id':authority.profile_id,'platform':'discord','inquiry_channels':list(inquiry),
        'coordinator_channels':list(coordinator),**(extras or {})}}))


async def run_native_inquiry(owner,monkeypatch,text='상태 확인',*,message_id='inquiry',
                             channel='42',actor='42',thread=None,parent=None,reply=None,
                             inquiry=('42',),coordinator=(),caption=True,images=(),extras=None):
    runner,adapter,authority = owner
    home = get_hermes_home()
    configure_binding(authority,inquiry,coordinator,extras)
    event = MessageEvent(text,source=adapter.build_source(chat_id=channel,chat_type='dm',user_id=actor,
        thread_id=thread,parent_chat_id=parent),message_id=message_id,timestamp=datetime.now(UTC),
        reply_to_message_id=reply,media_urls=[item.local_path for item in images],
        media_types=[item.mime for item in images])
    event._managed_platform_observation = PlatformObservation(caption,message_id,actor,tuple(images),reply)
    with native_callback(runner,event,home):
        receipt = await authority.admit_native(event)
    await authority._drain(receipt.ref)
    row = next(item for item in list_session_admissions(authority.db,session_id=receipt.ref.session_id,pending_only=False)
               if item['admission_id']==receipt.admission_id)
    with authority.db._read_ctx() as connection:
        terminal = connection.execute('SELECT value FROM state_meta WHERE key=?',
            (RESULT_PREFIX+receipt.admission_id,)).fetchone()
    result = load_result(terminal['value'])['result']
    return runner,adapter,authority,event,receipt,row,result


async def adapter_command(owner,text,message_id='command'):
    _runner,adapter,authority = owner
    event = MessageEvent(text,source=adapter.build_source(chat_id='42',chat_type='dm',user_id='42'),
                         message_id=message_id,timestamp=datetime.now(UTC))
    event._managed_platform_observation = PlatformObservation(True,message_id,'42',())
    task = asyncio.create_task(adapter.handle_message(event))
    row = None
    try:
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            with authority.db._read_ctx() as connection:
                row = connection.execute('SELECT * FROM session_admissions WHERE request_id=?',(message_id,)).fetchone()
            if row is not None: break
            await asyncio.sleep(.01)
        if row is not None:
            from gateway.session_contract import SessionRef
            await authority._drain(SessionRef(authority.profile_id,row['target_session_id']))
        await asyncio.wait_for(task,3)
        return row
    finally:
        if not task.done(): task.cancel()
        await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_real_native_fifo_run_turn_hook_captures_one_verified_completion(owner,monkeypatch):
    runner,adapter,_authority,_event,receipt,row,result = await run_native_inquiry(owner,monkeypatch)
    assert row['status'] == 'terminal'
    assert result['completed'] is True
    assert result['managed_inquiry_v1']['store_verified'] is True
    assert result['managed_inquiry_v1']['admission_id'] == receipt.admission_id
    assert result['managed_inquiry_v1']['admission_generation'] == row['generation']
    runner._hmwa_prepare_turn.assert_not_called()
    assert adapter.send.await_count == 1


@pytest.mark.asyncio
async def test_healthy_probe_and_new_admission_use_actual_b_core_each_time(owner,monkeypatch,external_io):
    first = await run_native_inquiry(owner,monkeypatch)
    assert first[-1]['managed_inquiry_v1']['state']=='health_verified'
    assert external_io.discoveries==len(external_io.gets)==1
    runner,_,authority,event,receipt,_,_ = first
    with native_callback(runner,event,get_hermes_home()):
        repeated = await authority.admit_native(event)
    await authority._drain(repeated.ref)
    assert repeated.admission_id==receipt.admission_id
    assert len(external_io.gets)==1
    again = await run_native_inquiry(owner,monkeypatch,message_id='new-challenge')
    assert again[-1]['managed_inquiry_v1']['state']=='health_verified'
    assert external_io.discoveries==len(external_io.gets)==2
    runner._hmwa_prepare_turn.assert_not_called()


@pytest.mark.asyncio
async def test_actual_timeout_then_late_healthy_probe_preserves_first_prepared_and_pending(owner,monkeypatch,external_io):
    from gateway import managed_inquiry_bridge as bridge
    from gateway.managed_inquiry import probe as probe_module
    from gateway.managed_inquiry.models import State
    monkeypatch.setattr(bridge,'INQUIRY_TIMEOUT_SECONDS',.15)
    actual = probe_module.classify
    classified = []
    def observe(*args):
        value = actual(*args)
        classified.append(value.state)
        return value
    monkeypatch.setattr(probe_module,'classify',observe)
    external_io.hold_close = True
    task = asyncio.create_task(run_native_inquiry(owner,monkeypatch))
    assert await asyncio.to_thread(external_io.closed.wait,2)
    result = await asyncio.wait_for(task,2)
    runner,adapter,_authority,_,_,_,terminal = result
    assert terminal['managed_inquiry_v1']['state']=='timed_out'
    event = adapter._native_test_events[-1]
    cap = event._managed_completion_capability
    assert cap.store.pending(cap.request.scope) is not None
    first = cap.store.cached(cap.request)
    external_io.release.set()
    deadline = time.monotonic()+2
    while State.HEALTHY not in classified and time.monotonic()<deadline:
        await asyncio.sleep(.01)
    assert State.HEALTHY in classified
    await asyncio.sleep(.05)
    assert cap.store.cached(cap.request)==first
    assert cap.store.pending(cap.request.scope) is not None
    assert adapter.send.await_count==1
    monkeypatch.setattr(bridge,'INQUIRY_TIMEOUT_SECONDS',90.)
    external_io.hold_close = False
    followup = await run_native_inquiry(owner,monkeypatch,message_id='fresh-after-timeout')
    assert followup[-1]['managed_inquiry_v1']['state']=='health_verified'
    assert len(external_io.gets)==2
    runner._hmwa_prepare_turn.assert_not_called()


def test_failed_send_restarts_in_fresh_process_and_real_ack_links_reply(tmp_path):
    import subprocess
    import os
    import sys
    home = tmp_path/'peer-home'
    home.mkdir()
    peer = Path(__file__).parent/'fixtures/managed_native_peer.py'
    env = {**os.environ,'HERMES_HOME':str(home)}
    receipts = []
    for phase in ('produce','restart'):
        child = subprocess.run([sys.executable,str(peer),phase],env=env,cwd=Path(__file__).parents[2],
            stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=25,check=False)
        assert child.returncode==0,child.stdout+child.stderr
        receipts.append(json.loads(next(line.split('=',1)[1] for line in child.stdout.splitlines()
                                        if line.startswith('NATIVE_RECEIPT='))))
    assert receipts[0]['pid']!=receipts[1]['pid']
    assert receipts[0]['sends']==2 and receipts[0]['gets']==1
    assert receipts[1]['sends']==1 and receipts[1]['gets']==0
    assert receipts[1]['claimed']==receipts[1]['delivered']==1
    assert receipts[1]['replay_verified']==[True]
    assert receipts[1]['reply_linked'] is True
    assert receipts[1]['historical_epoch']<receipts[1]['current_epoch']


def test_fresh_gateway_control_measures_registered_callables_and_disk_replacement(tmp_path):
    import subprocess
    import shutil
    import os
    import sys
    root = Path(__file__).resolve().parents[2]
    clone = tmp_path/'public-source'
    shutil.copytree(root,clone,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','.venv','.hermes','.env','node_modules'))
    home = tmp_path/'control-home'
    home.mkdir()
    child = subprocess.run([sys.executable,str(clone/'tests/gateway/fixtures/managed_native_peer.py'),'control'],
        env={**os.environ,'HERMES_HOME':str(home)},cwd=clone,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,timeout=25,check=False)
    assert child.returncode==0,child.stdout+child.stderr
    receipt = json.loads(next(line.split('=',1)[1] for line in child.stdout.splitlines()
                             if line.startswith('NATIVE_RECEIPT=')))
    assert receipt['pid']!=os.getpid()
    assert receipt['sends']==receipt['gets']==receipt['discoveries']==0
    before,after = receipt['before'],receipt['after']
    assert before['state']=='measured',before
    assert all(row['state']=='matched' and row['startup_equal'] for row in before['callables'].values())
    assert before['startup_source_set_sha256']==before['current_source_set_sha256']
    for name,row in before['callables'].items():
        expected = hashlib.sha256((root/row['source']).read_bytes()).hexdigest()
        assert row['source_sha256']==expected,name
    assert {'registered.0.handler','registered.0.probe','registered.0.renderer'}<=before['callables'].keys()
    assert before['not_covered']==['plugin_registration','hook_registration']
    assert after['state']=='incomplete'
    assert after['callables']['inquiry_handler']['state']=='loaded_disk_mismatch'
    assert after['startup_source_set_sha256']!=after['current_source_set_sha256']


async def historical_source(owner):
    from datetime import timedelta
    from hermes_state_runtime import claim_session_input,settle_session_input
    runner,adapter,authority = owner
    now = datetime.now(UTC)
    inputs = ('첫 번째 원본 요청의 전체 내용을 확인해주세요','이번에는 두 번째 전체 요청과 실제 실패를 조사해주세요')
    answer = '이것은 원본 요청에 대한 완전한 답변입니다. 흔한 화면 문구만으로 원본을 잘못 선택하지 않도록 충분히 긴 답변 전체를 그대로 기록합니다. 이어진 두 번째 요청에서 발생한 HTTP 실행 실패를 함께 확인해야 합니다.'
    sid = None
    for index,text in enumerate(inputs):
        event = MessageEvent(text,source=adapter.build_source(chat_id='7',chat_type='dm',user_id='42'),
            message_id='historical-'+str(index),timestamp=now-timedelta(seconds=300-index*100),
            auto_skill=['bluemango-ops-inquiry'])
        event._managed_platform_observation = PlatformObservation(True,'historical-'+str(index),'42',())
        with native_callback(runner,event,get_hermes_home()):
            receipt = await authority.admit_native(event)
        sid = receipt.ref.session_id
        row = claim_session_input(authority.db,epoch=authority.epoch,session_id=sid)
        assert isinstance(row,dict)
        settle_session_input(authority.db,epoch=authority.epoch,admission_id=receipt.admission_id,
                             generation=row['generation'],outcome='completed')
    calls = json.dumps([{'id':'historical-http','type':'function','function':{'name':'terminal',
        'arguments':json.dumps({'command':'curl https://fixture.invalid/healthz'})}}])
    assert sid is not None
    for role,content,stamp,call_id,tool_calls,tool_name in (
        ('assistant',answer,now.timestamp()-250,None,None,None),
        ('assistant','',now.timestamp()-190,None,calls,None),
        ('tool','JSONDecodeError\n<html><title>Moved</title></html>',now.timestamp()-180,'historical-http',None,'terminal')):
        authority.db.append_message(sid,role,content,timestamp=stamp,tool_call_id=call_id,
                                    tool_calls=json.loads(tool_calls) if tool_calls else None,tool_name=tool_name)
    helper = get_hermes_home()/'skills/bluemango-ops-inquiry/scripts/ops_status_probe.py'
    helper.parent.mkdir(parents=True,exist_ok=True)
    helper.write_text('fixture_helper = True\n')
    return sid,inputs[0]+answer+inputs[1]


@pytest.mark.asyncio
async def test_captionless_image_links_different_inquiry_channel_by_real_source_reader(owner,monkeypatch,external_io,tmp_path):
    import sys
    from gateway.managed_ingress_provenance import AttachmentInput
    runner,adapter,_authority = owner
    sid,hint = await historical_source(owner)
    executable = tmp_path/'fixture-ocr'
    payload = json.dumps({'lines':[{'candidates':[{'text':hint}]}]})
    executable.write_text('#!'+sys.executable+'\nprint('+repr(payload)+')\n')
    executable.chmod(0o700)
    att = SimpleNamespace(id='screen',filename='screen.png',content_type='image/png',size=40,
        url='https://invalid.example/image',read=AsyncMock(return_value=b'\x89PNG\r\n\x1a\n'+b'x'*32))
    images = []
    await adapter._collect_attachment_media([AttachmentInput(att,'current','image-now')],observations=images)
    result = await run_native_inquiry(owner,monkeypatch,'[Image attachment]',message_id='image-now',
        inquiry=('7',),coordinator=('42',),caption=False,images=images,
        extras={'ocr_executable':str(executable),'ocr_sha256':hashlib.sha256(executable.read_bytes()).hexdigest()})
    assert result[-1]['managed_inquiry_v1']['state']=='inspected'
    cap = adapter._native_test_events[-1]._managed_completion_capability
    assert cap.store.pending(cap.request.scope) is None
    snapshot = cap.store.lookup_delivery_completion(cap.prepared.completion_id,cap.request.scope,cap.prepared.text)
    assert snapshot.source.source_session_id==sid
    assert len(snapshot.source.source_admissions)==2
    assert len(external_io.gets)==1
    runner._hmwa_prepare_turn.assert_not_called()


@pytest.mark.asyncio
async def test_managed_slash_uses_actual_adapter_and_existing_native_fifo(owner,monkeypatch,external_io):
    runner,adapter,authority,_,_,_,_ = await run_native_inquiry(owner,monkeypatch)
    event = MessageEvent('/inquiry status',source=adapter.build_source(chat_id='42',chat_type='dm',user_id='42'),
                         message_id='slash',timestamp=datetime.now(UTC))
    event._managed_platform_observation = PlatformObservation(True,'slash','42',())
    task = asyncio.create_task(adapter.handle_message(event))
    try:
        deadline = time.monotonic()+3
        row = None
        while time.monotonic()<deadline:
            with authority.db._read_ctx() as connection:
                stored = connection.execute('SELECT target_session_id FROM session_admissions WHERE request_id="slash"').fetchone()
            if stored is not None:
                row = stored
                break
            await asyncio.sleep(.01)
        assert row is not None,'managed slash never entered native FIFO'
        from gateway.session_contract import SessionRef
        await authority._drain(SessionRef(authority.profile_id,row['target_session_id']))
        await asyncio.wait_for(task,3)
        assert len(external_io.gets)==2
        runner._hmwa_prepare_turn.assert_not_called()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('case',['captionless','typed_placeholder','ui_caption','reply','snapshot',
    'handoff_malformed','handoff_stale','handoff_other_waiter','handoff_text','handoff_source','handoff_command'])
async def test_raw_discord_producer_to_native_source_inspection(owner,monkeypatch,external_io,tmp_path,case):
    import sys
    from gateway.session_contract import SessionRef
    runner,adapter,authority = owner
    handoff_rejections=[]
    if case.startswith('handoff_'):
        from gateway import managed_inquiry_bridge as bridge
        actual_handoff=bridge.handoff_managed_completion
        def reject_handoff(current,admission_id,executed,waiter):
            from copy import copy
            if case=='handoff_malformed': executed._managed_completion_capability={}
            elif case=='handoff_stale':
                cap=executed._managed_completion_capability
                executed._managed_completion_capability=replace(cap,execution=replace(cap.execution,
                    admission_generation=cap.execution.admission_generation+1))
            elif case=='handoff_other_waiter': waiter=copy(waiter)
            elif case=='handoff_text': waiter.text+='changed'
            elif case=='handoff_source': waiter.source=replace(waiter.source,user_id='43')
            else: waiter.allow_gateway_control=not waiter.allow_gateway_control
            accepted=actual_handoff(current,admission_id,executed,waiter)
            handoff_rejections.append(accepted)
            return accepted
        monkeypatch.setattr(bridge,'handoff_managed_completion',reject_handoff)
    sid,hint = await historical_source(owner)
    executable=tmp_path/'raw-fixture-ocr'
    payload=json.dumps({'lines':[{'candidates':[{'text':hint}]}]})
    executable.write_text('#!'+sys.executable+'\nprint('+repr(payload)+')\n')
    executable.chmod(0o700)
    configure_binding(authority,('7',),('42',),{'ocr_executable':str(executable),
        'ocr_sha256':hashlib.sha256(executable.read_bytes()).hexdigest()})
    class DM:
        id=42
        name='fixture'
    monkeypatch.setattr('plugins.platforms.discord.adapter.discord.DMChannel',DM)
    adapter._client=SimpleNamespace(user=SimpleNamespace(id=999))
    att=SimpleNamespace(id=123,filename='current.png',content_type='image/png',size=40,
        url='https://invalid.example/image',read=AsyncMock(return_value=b'\x89PNG\r\n\x1a\n'+b'x'*32))
    caption={'typed_placeholder':'[Image attachment]','ui_caption':'UI 버튼의 색깔을 바꿔주세요'}.get(case,'')
    referenced=SimpleNamespace(content='',attachments=[att])
    message=SimpleNamespace(id=456,content=caption,attachments=[] if case in {'reply','snapshot'} else [att],
        mentions=[],reference=SimpleNamespace(message_id=455,resolved=referenced) if case=='reply' else None,
        message_snapshots=[SimpleNamespace(content='',attachments=[att])] if case=='snapshot' else [],
        channel=DM(),author=SimpleNamespace(id=42,display_name='Fixture',name='Fixture',bot=False),
        created_at=datetime.now(UTC))
    assert await adapter._handle_message(message)
    stored=None
    deadline=time.monotonic()+3
    while stored is None and time.monotonic()<deadline:
        with authority.db._read_ctx() as connection:
            stored=connection.execute('SELECT * FROM session_admissions WHERE request_id=?',('456',)).fetchone()
        if stored is None: await asyncio.sleep(.01)
    assert stored is not None,'actual Discord producer never entered native FIFO'
    await authority._drain(SessionRef(authority.profile_id,stored['target_session_id']))
    with authority.db._read_ctx() as connection:
        terminal=connection.execute('SELECT value FROM state_meta WHERE key=?',(RESULT_PREFIX+stored['admission_id'],)).fetchone()
    result=load_result(terminal['value'])['result']
    proof=json.loads(stored['payload_json'])['native_text_v1']['platform_ingress_v1']
    assert proof['caption_present'] is bool(caption)
    assert proof['attachments'][0]['origin']==('current' if case not in {'reply','snapshot'} else case)
    if case=='captionless' or case.startswith('handoff_'):
        assert result['managed_inquiry_v1']['state']=='inspected'
        await asyncio.wait_for(external_io.finalized.wait(),3)
        if case.startswith('handoff_'):
            assert handoff_rejections==[False]
            assert len(external_io.sends)==1
            with sqlite3.connect(get_hermes_home()/'managed-inquiry/state.sqlite') as connection:
                assert connection.execute('SELECT count(*) FROM delivery_refs').fetchone()[0]==0
            return
        with authority.db._read_ctx() as connection:
            key=connection.execute('SELECT managed_completion_key FROM delivery_obligations WHERE chat_id=?',('42',)).fetchone()
        with sqlite3.connect(get_hermes_home()/'managed-inquiry/state.sqlite') as connection:
            references=connection.execute('SELECT count(*) FROM delivery_refs').fetchone()[0]
        assert (key is not None and bool(key['managed_completion_key']),references)==(True,1)
        caps=[event._managed_completion_capability for event in adapter._native_test_events
              if hasattr(event,'_managed_completion_capability')]
        assert len(caps)==1
        cap=caps[0]
        snapshot=cap.store.lookup_delivery_completion(cap.prepared.completion_id,cap.request.scope,cap.prepared.text)
        assert snapshot.source.source_session_id==sid
        assert len(external_io.gets)==1
        assert len(external_io.sends)==1
        runner._hmwa_prepare_turn.assert_not_called()
    else:
        assert len(external_io.gets)==external_io.discoveries==0
        assert 'managed_inquiry_v1' not in result


@pytest.mark.asyncio
async def test_claimed_store_failure_is_typed_completion_without_agent_fallback(owner,monkeypatch):
    from gateway.managed_inquiry.task_state import PendingStore
    monkeypatch.setattr(PendingStore,'begin',lambda *args:(_ for _ in ()).throw(OSError('fixture_store_failure')))
    runner,adapter,_authority,_event,_receipt,row,result = await run_native_inquiry(owner,monkeypatch)
    assert row['status'] == 'terminal'
    assert result['managed_inquiry_v1']['store_verified'] is False
    assert result['managed_inquiry_v1']['state'] == 'handler_blocked'
    runner._hmwa_prepare_turn.assert_not_called()
    assert adapter.send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('tamper',['missing','forged','profile','actor','channel','thread','text','marker',
                                 'source_generation','source_epoch','source_payload','source_principal',
                                 'connector','route','terminal','finaltext','receipt','reset'])
async def test_replay_key_never_substitutes_for_original_native_and_b_receipts(owner,monkeypatch,external_io,tamper):
    from gateway import delivery_ledger as ledger
    from gateway.managed_inquiry_bridge import record_replayed_delivery
    from gateway.platforms.base import SendResult
    external_io.send_success = False
    runner,_,authority,_,receipt,_,_ = await run_native_inquiry(owner,monkeypatch)
    claimed = ledger.sweep_failed_for_runtime('discord',now=time.time()+60)
    assert len(claimed)==1
    row = claimed[0]
    locator = json.loads(row['managed_completion_key'])
    if tamper=='missing': row.pop('managed_completion_key')
    elif tamper=='forged': row['managed_completion_key']='{}'
    elif tamper=='profile': row['profile']='foreign'
    elif tamper in {'actor','channel','thread'}:
        field = {'actor':'actor_id','channel':'channel_id','thread':'thread_id'}[tamper]
        locator['scope'][field]='foreign'
        row['managed_completion_key']=json.dumps(locator)
        if tamper=='channel': row['chat_id']='foreign'
        if tamper=='thread': row['thread_id']='foreign'
    elif tamper=='text': row['content']+='tampered'
    elif tamper=='marker': row.update(needs_marker=True,marker='unapproved recovery marker')
    elif tamper in {'source_generation','source_epoch','source_principal'}:
        if tamper=='source_principal':
            authority.db._execute_write(lambda connection:connection.execute(
                'UPDATE session_admissions SET principal_id=? WHERE admission_id=?',('foreign',receipt.admission_id)))
        else:
            column = 'generation' if tamper=='source_generation' else 'owner_epoch'
            authority.db._execute_write(lambda connection:connection.execute(
                f'UPDATE session_admissions SET {column}={column}+1 WHERE admission_id=?',(receipt.admission_id,)))
    elif tamper in {'source_payload','connector','route'}:
        def corrupt(connection):
            from hermes_state_keys import admission_fingerprint
            stored=connection.execute('SELECT * FROM session_admissions WHERE admission_id=?',(receipt.admission_id,)).fetchone()
            payload=json.loads(stored['payload_json'])
            if tamper=='source_payload': payload['text']='changed after completion'
            elif tamper=='connector': payload['native_text_v1']['provenance']['connector']='foreign'
            else: payload['native_text_v1']['route']='foreign'
            digest=stored['payload_digest'] if tamper=='source_payload' else admission_fingerprint(
                canonical_target=stored['target_session_id'],payload={'input':payload,'intent':stored['intent']})
            connection.execute('UPDATE session_admissions SET payload_json=?,payload_digest=? WHERE admission_id=?',
                               (json.dumps(payload),digest,receipt.admission_id))
        authority.db._execute_write(corrupt)
    elif tamper in {'terminal','finaltext','receipt'}:
        def corrupt_terminal(connection):
            stored=connection.execute('SELECT value FROM state_meta WHERE key=?',(RESULT_PREFIX+receipt.admission_id,)).fetchone()
            terminal=load_result(stored['value'])
            if tamper=='terminal': terminal['result']['completed']=False
            elif tamper=='finaltext': terminal['result']['final_response']='foreign'
            else: terminal['result']['managed_inquiry_v1']['receipt_sha256']='0'*64
            connection.execute('UPDATE state_meta SET value=? WHERE key=?',
                               (json.dumps(terminal),RESULT_PREFIX+receipt.admission_id))
        authority.db._execute_write(corrupt_terminal)
    else:
        assert await adapter_command(owner,'/inquiry reset',message_id='reset') is not None
    content = (row.get('marker',ledger.RECOVERED_MARKER) if row.get('needs_marker') else '')+row['content']
    assert record_replayed_delivery(runner,row,content,SendResult(success=True,message_id='forged-ack')) is False
    with sqlite3.connect(get_hermes_home()/'managed-inquiry/state.sqlite') as connection:
        assert connection.execute('SELECT count(*) FROM delivery_refs').fetchone()[0]==0


@pytest.mark.asyncio
@pytest.mark.parametrize('case',['caption','generic_failure','foreign_profile','child_override','cross_actor'])
async def test_unselected_native_scope_and_ui_inputs_perform_no_ops(owner,monkeypatch,external_io,case):
    text = 'UI 버튼의 색깔을 바꿔주세요' if case!='generic_failure' else '일반적인 오류가 생겼어요'
    result = await run_native_inquiry(owner,monkeypatch,text,
        channel='9' if case=='child_override' else '42',
        thread='9' if case=='child_override' else None,
        parent='42' if case=='child_override' else None,
        inquiry=('42',) if case=='child_override' else (),
        coordinator=('9',) if case=='child_override' else ('42',),
        actor='43' if case=='cross_actor' else '42',
        extras={'profile_id':'foreign'} if case=='foreign_profile' else None)
    assert 'managed_inquiry_v1' not in result[-1]
    assert len(external_io.gets)==external_io.discoveries==0
    assert not (get_hermes_home()/'managed-inquiry/state.sqlite').exists()


@pytest.mark.asyncio
async def test_actual_unknown_slash_keeps_bypass_narrow(owner,external_io):
    _runner,_,authority = owner
    configure_binding(authority)
    assert await adapter_command(owner,'/unknown-unregistered') is None
    assert len(external_io.gets)==0


@pytest.mark.asyncio
async def test_advice_does_no_ops_and_parent_binding_inherits(owner,monkeypatch,external_io):
    runner,_,authority = owner
    configure_binding(authority)
    assert await adapter_command(owner,'/inquiry advice',message_id='advice') is not None
    assert len(external_io.gets)==0
    assert not (get_hermes_home()/'managed-inquiry/state.sqlite').exists()
    runner._hmwa_prepare_turn.reset_mock()
    result = await run_native_inquiry(owner,monkeypatch,message_id='inherited',channel='9',thread='9',parent='42')
    assert result[-1]['managed_inquiry_v1']['state']=='health_verified'
    assert len(external_io.gets)==1
    runner._hmwa_prepare_turn.assert_not_called()
