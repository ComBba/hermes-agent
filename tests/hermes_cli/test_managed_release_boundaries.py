import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import managed_release as policy


@pytest.mark.parametrize('record', [None, '{', '{}', json.dumps({'schema_version': 1, 'release_id': policy.MANAGED_RELEASE_ID})])
def test_public_contract_denies_missing_malformed_or_present_activation(tmp_path, monkeypatch, record):
    (tmp_path / 'hermes_cli').mkdir()
    (tmp_path / 'hermes_cli/managed_release.py').write_text('public contract')
    if record is not None:
        path = tmp_path / policy.ACTIVATION_RECORD
        path.parent.mkdir()
        path.write_text(record)
    monkeypatch.setenv('HERMES_MANAGED_RELEASE', '0')
    monkeypatch.setenv('HERMES_SKIP_UPDATE_CHECK', '1')
    with pytest.raises(policy.ManagedReleaseUpdateBlocked) as caught:
        policy.require_unmanaged_update(tmp_path, 'forced')
    assert caught.value.receipt.reason_code == 'managed_release_requires_governed_upgrade'
    assert caught.value.receipt.mutation_allowed is False
    assert str(tmp_path) not in str(caught.value)


def test_unmanaged_source_and_loaded_contract_identity(tmp_path, monkeypatch):
    policy.require_unmanaged_update(tmp_path, 'ordinary')
    monkeypatch.setattr(policy, 'SOURCE_ROOT', tmp_path)
    with pytest.raises(policy.ManagedReleaseUpdateBlocked):
        policy.require_unmanaged_update(tmp_path, 'marker_deleted')


@pytest.mark.parametrize('kind',[
    pytest.param('fifo',marks=pytest.mark.platforms('posix')),
    'directory','symlink','oversize'])
def test_nonregular_or_oversize_activation_is_denied_without_read(tmp_path,monkeypatch,kind):
    (tmp_path/'hermes_cli').mkdir()
    (tmp_path/'hermes_cli/managed_release.py').write_text('public contract')
    marker=tmp_path/policy.ACTIVATION_RECORD
    marker.parent.mkdir()
    if kind=='fifo': os.mkfifo(marker,0o600)
    elif kind=='directory': marker.mkdir()
    elif kind=='symlink':
        target=tmp_path/'record'
        target.write_text('{}')
        marker.symlink_to(target)
    else: marker.write_bytes(b' '*65537)
    original=Path.read_text
    reads=[]
    def trapped_read(path,*args,**kwargs):
        if path==marker:
            reads.append('activation_read')
            raise AssertionError('malformed activation reached read')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',trapped_read)
    with pytest.raises(policy.ManagedReleaseUpdateBlocked) as caught:
        policy.require_unmanaged_update(tmp_path,'forced')
    assert caught.value.receipt.activation_state=='malformed'
    assert caught.value.receipt.mutation_allowed is False
    assert reads==[]


@pytest.mark.parametrize('entry', ['run', 'prepare', 'bootstrap', 'selected', 'source', 'locked', 'finish'])
def test_each_mutation_entry_denies_before_any_downstream_action(monkeypatch, tmp_path, entry):
    from hermes_cli import update_completion as completion, source_completion as source, update_finish as finish
    root = policy.SOURCE_ROOT
    calls = []
    def mutation(*args, **kwargs):
        calls.append('mutation')
        raise AssertionError('managed update reached mutation')
    monkeypatch.setattr(completion.subprocess, 'run', mutation)
    monkeypatch.setattr(completion.subprocess, 'Popen', mutation)
    monkeypatch.setattr(completion.tempfile, 'TemporaryDirectory', mutation)
    monkeypatch.setattr(completion, '_record_owed_user_action', mutation)
    monkeypatch.setattr(completion, '_resume_receipt', mutation)
    if entry == 'bootstrap':
        monkeypatch.setattr(completion, '_prepare', mutation)
    from hermes_cli import source_build, update_cmd_maint, venv_sync, update_lock
    monkeypatch.setattr(source_build, 'build_update_products', mutation)
    monkeypatch.setattr(update_cmd_maint, '_run_post_update_maintenance', mutation)
    monkeypatch.setattr(venv_sync, 'publish_launchers', mutation)
    monkeypatch.setattr(update_lock.UpdateLock, 'acquire', mutation)
    before = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*')}
    request = {'source': str(root), 'home': str(tmp_path)}
    source_args = dict(desktop=False, assume_yes=True, gateway_mode=False,
                       pre_update_snapshot_id=None, pre_update_version=None, completion_message=None, announce=None)
    entries = {
        'run': lambda: completion.run_completion(request),
        'prepare': lambda: completion._prepare(request, tmp_path/'request', tmp_path/'result'),
        'bootstrap': lambda: completion._bootstrap(request, tmp_path/'request', tmp_path/'result'),
        'selected': lambda: completion._complete_selected(request),
        'source': lambda: source.complete_source_checkout(root, **source_args),
        'locked': lambda: source._complete_locked(root, **source_args),
        'finish': lambda: finish.finish_update(root=root, assume_yes=True, gateway_mode=False,
            pre_update_snapshot_id=None, had_desktop_app_before_update=False, pre_update_version=None,
            plan=None, windows_resume=None),
    }
    with pytest.raises(policy.ManagedReleaseUpdateBlocked):
        entries[entry]()
    assert calls == []
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*')} == before


def test_real_command_readonly_check_and_force_mutation_boundary(monkeypatch):
    from hermes_cli import main
    command = Mock(return_value=True)
    monkeypatch.setattr(main, '_update_preflight_handled', command)
    monkeypatch.setattr('hermes_cli.update_owning_install.retarget_to_owning_install', lambda root: None)
    assert main.cmd_update(SimpleNamespace(check=True)) is None
    command.assert_called_once()
    with pytest.raises(policy.ManagedReleaseUpdateBlocked):
        main.cmd_update(SimpleNamespace(check=False, force=True, yes=True))
    assert command.call_count == 1


@pytest.mark.parametrize('prepared',[False,True])
@pytest.mark.parametrize('entry',['completion_entry','source_entry','finish_entry'])
def test_child_entry_denies_before_dependency_activation_or_receipt(monkeypatch,tmp_path,entry,prepared):
    import sys
    from hermes_cli import update_completion,source_completion,update_finish,update_receipt
    import pm.environments
    calls = []
    def mutation(*args,**kwargs):
        calls.append('mutation')
        raise AssertionError('managed child performed mutation')
    monkeypatch.setattr(pm.environments,'activate_dependencies',mutation)
    monkeypatch.setattr(update_receipt,'begin_update_receipt',mutation)
    root = policy.SOURCE_ROOT
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'schema':1,'source':str(root),'root':str(root)}))
    arguments = ['completion',str(request),str(tmp_path/'result')]
    if prepared:
        arguments.append('--prepared')
    monkeypatch.setattr(sys,'argv',arguments)
    with pytest.raises(policy.ManagedReleaseUpdateBlocked):
        if entry == 'completion_entry':
            update_completion.main()
        elif entry == 'source_entry':
            source_completion.main(['--source',str(root),*(['--prepared'] if prepared else [])])
        else:
            update_finish.main(request,tmp_path/'result')
    assert calls == []
