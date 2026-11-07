from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import CodeType
from typing import Any


def _code_digest(code: CodeType) -> str:
    return hashlib.sha256(json.dumps(_code_value(code), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _code_value(value):
    if isinstance(value, CodeType):
        return {name: _code_value(getattr(value, name)) for name in (
            'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
            'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names',
            'co_varnames', 'co_filename', 'co_name', 'co_qualname', 'co_firstlineno',
            'co_linetable', 'co_exceptiontable', 'co_freevars', 'co_cellvars')}
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, tuple):
        return ['tuple', *(_code_value(item) for item in value)]
    if isinstance(value, frozenset):
        return ['frozenset', *sorted((_code_value(item) for item in value), key=repr)]
    return [type(value).__name__, repr(value)]


def _find_code(code: CodeType, name: str) -> CodeType | None:
    if code.co_qualname == name:
        return code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            found = _find_code(value, name)
            if found is not None:
                return found
    return None


def _measure(target, root: Path) -> dict[str, Any]:
    function = getattr(target, '__func__', target)
    code = getattr(function, '__code__', None)
    module = inspect.getmodule(function)
    module_file = getattr(module, '__file__', None)
    if not isinstance(code, CodeType) or module is None or not isinstance(module_file, str):
        return {'state': 'missing'}
    try:
        path = Path(code.co_filename).resolve(strict=True)
        module_path = Path(module_file).resolve(strict=True)
        if path != module_path or not path.is_relative_to(root):
            return {'state': 'origin_mismatch'}
        with path.open('rb') as handle:
            source = handle.read(2 * 1024 * 1024 + 1)
        if len(source) > 2 * 1024 * 1024:
            return {'state': 'source_oversize'}
        disk_code = _find_code(compile(source, code.co_filename, 'exec', dont_inherit=True,
                                      optimize=sys.flags.optimize), code.co_qualname)
        fingerprint = _code_digest(code)
        return {'state': 'matched' if disk_code is not None and _code_digest(disk_code) == fingerprint
                else 'loaded_disk_mismatch', 'source': str(path.relative_to(root)),
                'module': module.__name__, 'qualname': code.co_qualname,
                'source_sha256': hashlib.sha256(source).hexdigest(), 'code_sha256': fingerprint}
    except (OSError, SyntaxError, ValueError):
        return {'state': 'unreadable'}


def _targets(runner) -> dict[str, Any]:
    targets = {'foreground': getattr(runner, '_handle_message_with_agent', None),
               'background': getattr(runner, '_run_background_task_inner', None)}
    for name, module_name, attr in (
            ('reasoning_resolver', 'gateway.channel_reasoning', 'resolve_turn_reasoning'),
            ('verified_ingress', 'gateway.managed_ingress_provenance', 'verified_ingress'),
            ('inquiry_handler', 'gateway.managed_inquiry_bridge', 'handle_turn')):
        module = sys.modules.get(module_name)
        targets[name] = getattr(module, attr, None) if module is not None else None
    registry = getattr(runner,'_managed_inquiry_services',None)
    if isinstance(registry,dict):
        if len(registry)>8:
            targets['service_registry_overflow'] = None
        else:
            for index,entry in enumerate(registry.values()):
                if not isinstance(entry,tuple) or len(entry)!=2:
                    targets[f'registered.{index}.invalid'] = None
                    continue
                services = entry[1]
                prefix = f'registered.{index}.'
                targets[prefix+'probe_callback'] = getattr(services,'probe',None)
                source = getattr(services,'source',None)
                targets[prefix+'source_find'] = getattr(source,'find',None)
                targets[prefix+'source_inspect'] = getattr(source,'inspect',None)
                targets[prefix+'state_verify'] = getattr(getattr(services,'state',None),'verify',None)
                module = sys.modules.get('gateway.managed_inquiry.handler')
                handler = getattr(module,'handle',None) if module is not None else None
                targets[prefix+'handler'] = handler
                globals_ = getattr(handler,'__globals__',{})
                targets[prefix+'renderer'] = globals_.get('render_probe')
                callback = getattr(services,'probe',None)
                code = getattr(callback,'__code__',None)
                closure = getattr(callback,'__closure__',None)
                cells = dict(zip(code.co_freevars,closure)) if code is not None and closure is not None else {}
                targets[prefix+'probe'] = cells['probe'].cell_contents if 'probe' in cells else None
    return targets


def _source_set(callables) -> str:
    return hashlib.sha256(json.dumps(callables,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seal_loaded_receipt(runner) -> None:
    from gateway.managed_inquiry_bridge import registration_bindings
    root = Path(__file__).resolve().parents[1]
    runner._managed_loaded_baseline = {
        'pid': os.getpid(), 'root': root,
        'callables': {name: _measure(target, root) for name, target in _targets(runner).items()},
        'bindings': registration_bindings(runner),
    }


def loaded_receipt(runner) -> dict[str, Any]:
    from gateway.managed_inquiry_bridge import registration_bindings
    baseline = getattr(runner, '_managed_loaded_baseline', None)
    if not isinstance(baseline, dict) or baseline.get('pid') != os.getpid():
        return {'schema_version': 1, 'state': 'missing', 'not_covered': ['startup_measurement']}
    current = {name: _measure(target, baseline['root']) for name, target in _targets(runner).items()}
    bindings=registration_bindings(runner)
    results = {}
    for name, measurement in current.items():
        initial = baseline['callables'].get(name)
        results[name] = {**measurement, 'startup_equal': initial == measurement}
    covered = (set(current)==set(baseline['callables']) and bindings==baseline['bindings']
               and all(row['state'] in {'prepared','disabled'} for row in bindings.values())
               and all(row['state'] == 'matched' and row['startup_equal'] for row in results.values()))
    missing = ['plugin_registration','hook_registration']
    if not any(name.startswith('registered.') for name in results):
        missing += ['inquiry_services','probe_registration','renderer_registration']
    return {'schema_version': 1, 'pid': os.getpid(), 'state': 'measured' if covered else 'incomplete',
            'startup_source_set_sha256':_source_set(baseline['callables']),
            'startup_bindings':baseline['bindings'],'current_bindings':bindings,
            'startup_bindings_equal':bindings==baseline['bindings'],
            'current_source_set_sha256':_source_set(current),'callables': results,'not_covered':missing}
