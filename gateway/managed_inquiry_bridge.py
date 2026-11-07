from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from gateway.managed_ingress_provenance import ExecutionAdmission, execution_admission, verified_ingress

INQUIRY_TIMEOUT_SECONDS = 90.0

@dataclass(frozen=True, slots=True)
class _CompletionCapability:
    execution: ExecutionAdmission
    event_sha256: str
    text_sha256: str
    completion_id: str
    receipt_sha256: str
    prepared: Any = None
    request: Any = None
    store: Any = None


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _event_digest(event) -> str:
    source = event.source
    return _digest([source.to_dict(),event.message_id,event.timestamp.isoformat(),
                    event.text,event.reply_to_message_id])


def verified_managed_completion(event, text: str) -> bool:
    cap = getattr(event,'_managed_completion_capability',None)
    return (isinstance(cap,_CompletionCapability) and isinstance(text,str)
            and execution_admission(event,session_id=cap.execution.session_id) == cap.execution
            and _event_digest(event) == cap.event_sha256 and _digest(text) == cap.text_sha256)


def handoff_managed_completion(authority,admission_id: str,executed,waiter) -> bool:
    cap = getattr(executed,'_managed_completion_capability',None)
    if not isinstance(cap,_CompletionCapability) or cap.prepared is None:
        return False
    try:
        text = cap.prepared.text
        owner = cap.execution
        if (not verified_managed_completion(executed,text)
                or cap.store is None or not cap.store.verify(cap.prepared,cap.request)
                or admission_id != owner.admission_id or authority.profile_id != owner.profile_id
                or authority.epoch != owner.authority_epoch
                or getattr(authority,'marker_adopters',{}).get(admission_id) is not waiter
                or _event_digest(waiter) != cap.event_sha256
                or executed.allow_gateway_control != waiter.allow_gateway_control
                or executed.get_command() != waiter.get_command()
                or getattr(waiter,'_managed_completion_capability',None) is not None
                or getattr(waiter,'_managed_execution_admission',None) is not None
                or getattr(waiter,'_claimed_owner',None) is not None):
            return False
        with authority.db._read_ctx() as connection:
            row = connection.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                     (admission_id,)).fetchone()
        if (row is None or row['status'] != 'started' or row['generation'] != owner.admission_generation
                or row['owner_epoch'] != owner.authority_epoch or row['principal_id'] != owner.principal_id
                or row['request_id'] != owner.request_id
                or (row['target_session_id'] or row['session_id']) != owner.session_id):
            return False
        from hermes_state_keys import admission_fingerprint
        if admission_fingerprint(canonical_target=owner.session_id,
                payload={'input':json.loads(row['payload_json']),'intent':row['intent']}) != row['payload_digest']:
            return False
        waiter._managed_execution_admission = owner
        waiter._claimed_owner = (owner.session_id,owner.admission_generation)
        waiter._managed_completion_capability = cap
        return verified_managed_completion(waiter,text)
    except Exception:
        return False


def record_successful_delivery(event, text: str, result) -> bool:
    if (getattr(result,'success',None) is not True or not verified_managed_completion(event,text)):
        return False
    message_id = getattr(result,'message_id',None)
    if (not isinstance(message_id,str) or not message_id or len(message_id)>128
            or any(character.isspace() for character in message_id)):
        return False
    cap = event._managed_completion_capability
    if cap.prepared is None or cap.request is None or cap.store is None:
        return False
    try:
        if not cap.store.verify(cap.prepared,cap.request):
            return False
        cap.store.record_delivery_reference(cap.prepared,cap.request,message_id)
    except Exception:
        return False
    return True


def managed_completion_key(event,text: str) -> str | None:
    if not verified_managed_completion(event,text):
        return None
    cap = event._managed_completion_capability
    if cap.prepared is None or cap.request is None or cap.request.ingress is None:
        return None
    key = json.dumps({'schema_version':1,'runtime_home':cap.request.ingress.runtime_home,
        'completion_id':cap.completion_id,'receipt_sha256':cap.receipt_sha256,
        'scope':asdict(cap.request.scope),'text_sha256':_digest(text)},sort_keys=True,separators=(',',':'))
    return key if len(key)<=4096 else None


def record_replayed_delivery(runner,row,sent_text: str,result) -> bool:
    if getattr(result,'success',None) is not True:
        return False
    message_id = getattr(result,'message_id',None)
    key = row.get('managed_completion_key')
    if (not isinstance(message_id,str) or not message_id or len(message_id)>128
            or any(char.isspace() for char in message_id)
            or not isinstance(key,str) or not key or len(key)>4096):
        return False
    try:
        from gateway.delivery_ledger import RECOVERED_MARKER,RECONNECTED_MARKER,FLOOD_MARKER
        content = row['content']
        expected = content
        if row.get('needs_marker'):
            marker = row.get('marker',RECOVERED_MARKER)
            if marker not in {RECOVERED_MARKER,RECONNECTED_MARKER,FLOOD_MARKER}:
                return False
            expected = marker+content
        if sent_text != expected:
            return False
        locator = json.loads(key)
        if (not isinstance(locator,dict) or set(locator) != {'schema_version','runtime_home',
                'completion_id','receipt_sha256','scope','text_sha256'}
                or type(locator['schema_version']) is not int or locator['schema_version'] != 1
                or locator['text_sha256'] != _digest(content)):
            return False
        from gateway.session_authorities import authority_for_home
        from gateway.managed_inquiry.task_models import TaskScope
        from gateway.managed_inquiry.task_state import PendingStore
        from hermes_state_terminal import RESULT_PREFIX,load_result
        home = Path(locator['runtime_home'])
        if not home.is_absolute() or str(home.resolve()) != str(home):
            return False
        authority = authority_for_home(runner,home)
        if authority is None:
            return False
        scope = TaskScope(**locator['scope'])
        if (scope.profile_id != authority.profile_id or scope.platform != row['platform']
                or scope.channel_id != str(row['chat_id']) or scope.thread_id != str(row.get('thread_id') or '')):
            return False
        store = PendingStore(home/'managed-inquiry'/'state.sqlite')
        snapshot = store.lookup_delivery_completion(locator['completion_id'],scope,content)
        if snapshot is None or snapshot.completion.receipt_sha256 != locator['receipt_sha256']:
            return False
        completion = snapshot.completion
        binding = snapshot.binding
        with authority.db._read_ctx() as connection:
            native = connection.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                        (completion.admission_id,)).fetchone()
            terminal = connection.execute('SELECT value FROM state_meta WHERE key=?',
                                           (RESULT_PREFIX+completion.admission_id,)).fetchone()
        if native is None or terminal is None or native['status'] != 'terminal':
            return False
        payload = json.loads(native['payload_json'])
        from hermes_state_keys import admission_fingerprint
        target = native['target_session_id'] or native['session_id']
        if admission_fingerprint(canonical_target=target,
                payload={'input':payload,'intent':native['intent']}) != native['payload_digest']:
            return False
        envelope = payload.get('native_text_v1',{})
        source = envelope.get('source',{})
        from gateway.session_envelope import restore_native_source
        restored_source = restore_native_source(payload,runner)
        if runner._transport_owner(restored_source) is None:
            return False
        transport_profile = runner._adapter_profile_for_source(restored_source) or 'default'
        original = load_result(terminal['value'])
        final = original.get('result',{})
        recorded = final.get('managed_inquiry_v1',{})
        if (native['generation'] != completion.admission_generation
                or native['owner_epoch'] != completion.authority_epoch
                or native['principal_id'] != binding.execution.principal_id
                or native['request_id'] != binding.execution.request_id
                or envelope.get('event',{}).get('message_id') != binding.message_id
                or envelope.get('platform_ingress_v1',{}).get('binding_sha256') != binding.ingress_binding_sha256
                or envelope.get('provenance',{}).get('connector') != binding.connector
                or (native['target_session_id'] or native['session_id']) != scope.session_id
                or source.get('platform') != scope.platform or str(source.get('chat_id')) != scope.channel_id
                or str(source.get('thread_id') or '') != scope.thread_id or source.get('user_id') != scope.actor_id
                or source.get('profile') != scope.source_profile
                or envelope.get('route') != row['session_key']
                or transport_profile != row.get('profile')
                or final.get('final_response') != content or final.get('completed') is not True
                or recorded.get('completion_id') != completion.completion_id
                or recorded.get('receipt_sha256') != completion.receipt_sha256
                or recorded.get('store_verified') is not True):
            return False
        store.record_delivery_replay(snapshot,message_id)
        return True
    except Exception:
        return False


def _binding(runner,source,session_id):
    from gateway.session_identity import identity_of
    from gateway.session_authorities import authority_for_home
    from hermes_cli.config_effective import load_user_config_effective
    identity = identity_of(source)
    if identity is None:
        return None
    home = Path(identity.runtime_home).resolve()
    authority = authority_for_home(runner,home)
    if authority is None:
        return None
    config = load_user_config_effective(home/'config.yaml',fail_closed=True)
    section = config.get('managed_inquiry')
    if not isinstance(section,dict) or section.get('enabled') is not True:
        return None
    from gateway.managed_inquiry.bindings import resolve_binding
    from gateway.managed_inquiry.task_models import BindingRole, TaskScope
    scope = TaskScope(authority.profile_id,source.platform.value,str(source.chat_id),
                      str(source.thread_id or ''),str(source.user_id),session_id,
                      runtime_profile=identity.runtime_profile,source_profile=source.profile)
    role = resolve_binding(section,scope,getattr(source,'parent_chat_id',None))
    return None if role == BindingRole.NONE else (home,section,scope,role)


def managed_command_selected(runner,event,source) -> bool:
    if event.get_command() != 'inquiry' or event.internal:
        return False
    try:
        return _binding(runner,source,'') is not None
    except Exception:
        return True


def _services(runner,home,section,profile_id: str):
    from gateway.managed_inquiry.handler import InquiryServices
    from gateway.managed_inquiry.task_state import PendingStore
    from gateway.managed_inquiry.source import LocalOcr, SQLiteInquirySource
    from gateway.managed_inquiry.discovery import DiscoveryConfig, MacDiscovery
    from gateway.managed_inquiry.models import ServiceIdentity
    from gateway.managed_inquiry.probe import probe
    registry = getattr(runner,'_managed_inquiry_services',None)
    if registry is None:
        registry = runner._managed_inquiry_services = {}
    key = (profile_id,str(home))
    config_sha256 = _digest(section)
    entry = registry.get(key)
    if entry is not None and entry[0] == config_sha256:
        return entry[1]
    identity_config = section.get('service_identity')
    identity = None
    if isinstance(identity_config,dict) and set(identity_config) == {'field','value'}:
        identity = ServiceIdentity(**identity_config)
    executable,sha256 = section.get('ocr_executable'),section.get('ocr_sha256')
    ocr = LocalOcr(Path(executable),sha256) if isinstance(executable,str) and isinstance(sha256,str) else None
    services = InquiryServices(state=PendingStore(home/'managed-inquiry'/'state.sqlite'),
        probe=lambda control: probe(MacDiscovery(DiscoveryConfig(identity=identity),control=control),
                                    control=control),
        source=SQLiteInquirySource(home/'state.db',ocr=ocr))
    registry[key] = (config_sha256,services)
    return services


def _service_config(authority):
    from hermes_cli.config_effective import _effective
    from hermes_cli.config_backend import read_config_doc_readonly
    from gateway.managed_inquiry.bindings import _channels
    home=Path(authority.profile_id)
    if (not home.is_absolute() or home.resolve()!=home
            or Path(authority.db.db_path).resolve().parent!=home):
        raise ValueError('managed_profile_binding')
    try:
        raw=read_config_doc_readonly(home/'config.yaml')
    except FileNotFoundError:
        raw={}
    section=_effective(raw if isinstance(raw,dict) else {}).get('managed_inquiry')
    if section is None or (isinstance(section,dict) and section.get('enabled') is False):
        return home,section,'disabled'
    if (not isinstance(section,dict) or section.get('enabled') is not True
            or type(section.get('schema_version')) is not int or section['schema_version']!=1
            or section.get('profile_id')!=authority.profile_id or section.get('platform')!='discord'):
        return home,section,'invalid_config'
    inquiry,coordinator=_channels(section.get('inquiry_channels')),_channels(section.get('coordinator_channels'))
    return home,section,('valid' if inquiry is not None and coordinator is not None
                        and not set(inquiry)&set(coordinator) else 'invalid_config')


def prepare_services(runner) -> None:
    from importlib import import_module
    from gateway.session_authorities import serving_authorities
    for module in ('gateway.channel_reasoning','gateway.managed_ingress_provenance'):
        import_module(module)
    failures={}
    for authority in serving_authorities(runner)[:8]:
        key=(authority.profile_id,str(Path(authority.profile_id)))
        try:
            home,section,state=_service_config(authority)
            if state=='valid':
                _services(runner,home,section,authority.profile_id)
            else:
                registry=getattr(runner,'_managed_inquiry_services',{})
                registry.pop(key,None)
        except Exception:
            getattr(runner,'_managed_inquiry_services',{}).pop(key,None)
            failures[_digest(key)]='registration_failed'
    runner._managed_inquiry_preparation_failures=failures


def registration_bindings(runner):
    from gateway.session_authorities import serving_authorities
    authorities=serving_authorities(runner)
    observations={}
    if len(authorities)>8:
        observations['profile_limit']={'state':'overflow'}
    for authority in authorities[:8]:
        key=(authority.profile_id,str(Path(authority.profile_id)))
        opaque=_digest(key)
        try:
            home,section,state=_service_config(authority)
            entry=getattr(runner,'_managed_inquiry_services',{}).get(key)
            record={'state':state,'config_sha256':_digest(section),'registered_config_sha256':None}
            if state=='valid':
                record['state']='unregistered'
                if isinstance(entry,tuple) and len(entry)==2:
                    record['registered_config_sha256']=entry[0]
                    services=entry[1]
                    paths=(Path(services.state.path).resolve(),Path(services.source.db_path).resolve())
                    record['state']=('prepared' if entry[0]==record['config_sha256']
                        and paths==(home/'managed-inquiry/state.sqlite',home/'state.db') else 'binding_mismatch')
                elif opaque in getattr(runner,'_managed_inquiry_preparation_failures',{}):
                    record['state']='registration_failed'
            observations[opaque]=record
        except Exception:
            observations[opaque]={'state':'config_unreadable'}
    return observations


def _capture(event,execution,text,completion,prepared=None,request=None,store=None):
    from gateway.session_results import execution_result
    record = {'final_response':text,'messages':[],'completed':True,
              'managed_inquiry_v1':completion}
    captured = execution_result.get()
    if captured is not None:
        captured['result'] = record
    if execution is not None:
        event._managed_completion_capability = _CompletionCapability(execution,_event_digest(event),
            _digest(text),completion['completion_id'],completion['receipt_sha256'],prepared,request,store)
    return text


def _blocked(event,execution,reason):
    text = '관리형 문의를 안전하게 완료할 수 없습니다. 현재 작업은 차단 상태이며 다시 확인이 필요합니다.'
    completion = {'state':reason,'store_verified':False,'completion_id':_digest([
        asdict(execution) if execution is not None else None,reason]),'receipt_sha256':_digest(reason)}
    return _capture(event,execution,text,completion)


async def handle_turn(runner,event,source,session_entry,session_key,run_generation) -> str | None:
    execution = execution_admission(event,session_id=session_entry.session_id)
    try:
        binding = _binding(runner,source,session_entry.session_id)
    except Exception:
        return _blocked(event,execution,'binding_blocked')
    if binding is None:
        return None
    home,section,scope,role = binding
    if execution is None or execution.profile_id != scope.profile_id:
        return _blocked(event,execution,'ingress_blocked')
    request = None
    task = None
    services = None
    handler = None
    try:
        from gateway.managed_inquiry import handler
        from gateway.managed_inquiry.task_models import InquiryRequest,RequestControl
        check = verified_ingress(event,session_id=session_entry.session_id,
            admission_id=execution.admission_id,admission_generation=execution.admission_generation,
            authority_epoch=execution.authority_epoch)
        if check.state == 'invalid':
            return _blocked(event,execution,'ingress_blocked')
        request = InquiryRequest(scope,role,event.text,str(event.message_id or ''),execution,
            check.provenance,reply_to_message_id=event.reply_to_message_id,internal=event.internal,
            control=RequestControl(deadline=time.monotonic()+INQUIRY_TIMEOUT_SECONDS))
        services = _services(runner,home,section,scope.profile_id)
        task = asyncio.create_task(asyncio.to_thread(handler.handle,request,services))
        try:
            completion = await asyncio.wait_for(asyncio.shield(task),max(0,request.control.deadline-time.monotonic()))
        except TimeoutError:
            request.control.cancel()
            completion = await asyncio.to_thread(handler.timeout,request,services)
        if completion is None:
            return (_blocked(event,execution,'selection_contract_failed')
                    if request.control.expired else None)
        if (not services.state.verify(completion,request)
                or completion.admission_id != execution.admission_id
                or completion.admission_generation != execution.admission_generation
                or completion.authority_epoch != execution.authority_epoch
                or completion.scope_key != scope.key or completion.request_sha256 != request.identity_sha256):
            return _blocked(event,execution,'completion_receipt_invalid')
        return _capture(event,execution,completion.text,{**asdict(completion),'store_verified':True,
            'run_generation':run_generation},completion,request,services.state)
    except asyncio.CancelledError:
        if request is not None:
            request.control.cancel()
        raise
    except Exception:
        if request is not None:
            request.control.cancel()
            if services is not None and handler is not None:
                try:
                    await asyncio.to_thread(handler.failed,request,services)
                except Exception:
                    pass
        return _blocked(event,execution,'handler_blocked')
    finally:
        if task is not None:
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
