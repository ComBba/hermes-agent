import importlib.util
import sys
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.gateway import test_managed_inquiry_bridge as native
from types import SimpleNamespace

from gateway.managed_loaded_receipt import _measure, loaded_receipt, seal_loaded_receipt

owner = native.owner
external_io = native.external_io


def test_loaded_code_is_checked_against_disk_and_startup(tmp_path):
    path = tmp_path/'selected.py'
    path.write_text('def selected():\n    return "loaded"\n')
    spec = importlib.util.spec_from_file_location('managed_selected_test',path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        initial = _measure(module.selected,tmp_path)
        assert initial['state'] == 'matched'
        path.write_text('def selected():\n    return "replaced"\n')
        assert module.selected() == 'loaded'
        drift = _measure(module.selected,tmp_path)
        assert drift['state'] == 'loaded_disk_mismatch'
        assert drift['code_sha256'] == initial['code_sha256']
        assert drift['source_sha256'] != initial['source_sha256']
    finally:
        sys.modules.pop(spec.name,None)


def test_actual_gateway_callable_objects_are_measured_missing_services_are_explicit():
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig
    runner = GatewayRunner(GatewayConfig())
    assert loaded_receipt(runner)['state'] == 'missing'
    seal_loaded_receipt(runner)
    receipt = loaded_receipt(runner)
    assert receipt['callables']['foreground']['state'] == 'matched'
    assert receipt['callables']['background']['state'] == 'matched'
    assert receipt['callables']['foreground']['startup_equal'] is True
    assert 'probe_registration' in receipt['not_covered']


def test_actual_held_postconnect_cold_control_receipt_survives_first_selected_turn(tmp_path):
    root=Path(__file__).resolve().parents[2]
    home=tmp_path/'cold-home'
    home.mkdir()
    child=subprocess.run([sys.executable,str(root/'tests/gateway/fixtures/managed_native_peer.py'),'cold'],
        env={**os.environ,'HERMES_HOME':str(home)},cwd=root,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,timeout=25,check=False)
    assert child.returncode==0,child.stdout+child.stderr
    receipt=json.loads(next(line.split('=',1)[1] for line in child.stdout.splitlines() if line.startswith('NATIVE_RECEIPT=')))
    assert receipt['cold']=={'gets':0,'sends':0,'discoveries':0,'pending_db':False}
    before,after=receipt['before'],receipt['after']
    assert before['state']=='measured'
    assert after['state']=='measured'
    assert before['startup_source_set_sha256']==after['current_source_set_sha256']
    assert all(row['startup_equal'] for row in after['callables'].values())
    assert {'registered.0.handler','registered.0.probe','registered.0.renderer'}<=before['callables'].keys()
    assert before['not_covered']==after['not_covered']==['plugin_registration','hook_registration']
    assert receipt['gets']==receipt['sends']==receipt['discoveries']==1
    assert before['startup_bindings_equal'] is after['startup_bindings_equal'] is True
    assert before['current_bindings']==after['current_bindings']
    assert {row['state'] for row in before['current_bindings'].values()}=={'prepared'}


@pytest.mark.asyncio
async def test_postconnect_prepares_real_served_profiles_without_work_or_store_creation(owner,external_io):
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    runner,_adapter,authority=owner
    home=get_hermes_home()
    operator=home/'profiles/operator'
    operator.mkdir(parents=True)
    database=SessionDB(operator/'state.db')
    secondary=await initialize_session_authority(runner,profile_id=str(operator),instance_id='secondary',db=database,register=False)
    registry=SessionAuthorities(home,multiplexed=True)
    registry.add(home,authority)
    registry.add(operator,secondary,name='operator')
    runner.session_authorities=registry
    native.configure_binding(authority)
    (operator/'config.yaml').write_text(json.dumps({'managed_inquiry':{'schema_version':1,'enabled':True,
        'profile_id':str(operator),'platform':'discord','inquiry_channels':['other'],'coordinator_channels':[]}}))
    runner._external_drain_active=True
    try:
        await runner._start_finish_wiring(0)
        services=runner._managed_inquiry_services
        first=services[(authority.profile_id,str(home))][1]
        second=services[(secondary.profile_id,str(operator))][1]
        assert first is not second
        assert first.state.path==home/'managed-inquiry/state.sqlite'
        assert second.state.path==operator/'managed-inquiry/state.sqlite'
        assert not first.state.path.exists() and not second.state.path.exists()
        assert external_io.gets==external_io.sends==[] and external_io.discoveries==0
        for current in (authority,secondary):
            with current.db._read_ctx() as connection:
                assert connection.execute('SELECT count(*) FROM session_admissions').fetchone()[0]==0
        seal_loaded_receipt(runner)
        receipt=loaded_receipt(runner)
        assert receipt['state']=='measured'
        assert len(receipt['current_bindings'])==2
        assert all(row['state']=='prepared' for row in receipt['current_bindings'].values())
        await runner._start_finish_wiring(0)
        assert services[(authority.profile_id,str(home))][1] is first
        assert loaded_receipt(runner)['startup_bindings_equal'] is True
        services[(authority.profile_id,str(home))],services[(secondary.profile_id,str(operator))]=(
            services[(secondary.profile_id,str(operator))],services[(authority.profile_id,str(home))])
        drift=loaded_receipt(runner)
        assert drift['state']=='incomplete' and drift['startup_bindings_equal'] is False
        assert all(row['state']=='binding_mismatch' for row in drift['current_bindings'].values())
        services[(authority.profile_id,str(home))],services[(secondary.profile_id,str(operator))]=(
            services[(secondary.profile_id,str(operator))],services[(authority.profile_id,str(home))])
        (operator/'config.yaml').write_text(json.dumps({'managed_inquiry':{'schema_version':1,'enabled':True,
            'profile_id':'foreign','platform':'discord','inquiry_channels':['other'],'coordinator_channels':[]}}))
        await runner._start_finish_wiring(0)
        assert len(services)==1 and services[(authority.profile_id,str(home))][1] is first
        states={row['state'] for row in loaded_receipt(runner)['current_bindings'].values()}
        assert states=={'prepared','invalid_config'}
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('case',['disabled','foreign','schema','platform','channels','yaml','registration'])
async def test_cold_invalid_or_disabled_profile_does_not_register_or_create_work(owner,external_io,case):
    from hermes_constants import get_hermes_home
    runner,_adapter,authority=owner
    extras={'enabled':False} if case=='disabled' else {'profile_id':'foreign'} if case=='foreign' else (
        {'schema_version':True} if case=='schema' else {'platform':'api_server'} if case=='platform' else (
        {'inquiry_channels':['42','42']} if case=='channels' else (
        {'service_identity':{'field':'','value':''}} if case=='registration' else {})))
    native.configure_binding(authority,extras=extras)
    if case=='yaml': (get_hermes_home()/'config.yaml').write_text('{')
    runner._external_drain_active=True
    await runner._start_finish_wiring(0)
    seal_loaded_receipt(runner)
    receipt=loaded_receipt(runner)
    assert not getattr(runner,'_managed_inquiry_services',{})
    assert not (get_hermes_home()/'managed-inquiry/state.sqlite').exists()
    assert external_io.gets==external_io.sends==[] and external_io.discoveries==0
    with authority.db._read_ctx() as connection:
        assert connection.execute('SELECT count(*) FROM session_admissions').fetchone()[0]==0
    expected='disabled' if case=='disabled' else 'config_unreadable' if case=='yaml' else (
        'registration_failed' if case=='registration' else 'invalid_config')
    assert {row['state'] for row in receipt['current_bindings'].values()}=={expected}
    assert 'inquiry_services' in receipt['not_covered']
    if case!='disabled': assert receipt['state']=='incomplete'


@pytest.mark.asyncio
async def test_current_profile_config_drift_invalidates_startup_binding_without_warming(owner,external_io):
    runner,_adapter,authority=owner
    native.configure_binding(authority)
    runner._external_drain_active=True
    await runner._start_finish_wiring(0)
    seal_loaded_receipt(runner)
    original=loaded_receipt(runner)
    native.configure_binding(authority,inquiry=('different',))
    drift=loaded_receipt(runner)
    assert original['state']=='measured' and drift['state']=='incomplete'
    assert drift['startup_bindings_equal'] is False
    assert all(row['state']=='binding_mismatch' for row in drift['current_bindings'].values())
    assert all(row['startup_equal'] for row in drift['callables'].values())
    assert external_io.gets==external_io.sends==[] and external_io.discoveries==0


@pytest.mark.asyncio
async def test_warm_and_readonly_receipt_do_not_write_good_config_backups(owner,monkeypatch):
    from hermes_constants import get_hermes_home
    runner,_adapter,authority=owner
    native.configure_binding(authority)
    attempts=[]
    def forbidden_backup(*args,**kwargs):
        attempts.append('backup')
        raise AssertionError('readonly loaded preparation attempted backup')
    monkeypatch.setattr('hermes_cli.config_backups.backup_config',forbidden_backup)
    before={str(path.relative_to(get_hermes_home())) for path in get_hermes_home().rglob('*')}
    runner._external_drain_active=True
    await runner._start_finish_wiring(0)
    seal_loaded_receipt(runner)
    assert loaded_receipt(runner)['state']=='measured'
    native.configure_binding(authority,inquiry=('changed',))
    assert loaded_receipt(runner)['state']=='incomplete'
    assert attempts==[]
    assert {str(path.relative_to(get_hermes_home())) for path in get_hermes_home().rglob('*')}==before
