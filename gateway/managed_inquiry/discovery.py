from __future__ import annotations

import ast
import hashlib
import json
import os
import plistlib
import re
import shlex
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

from .models import (
    Discovery,
    JsonValue,
    ProbeControl,
    ServiceIdentity,
    Sha256,
    State,
    TargetEvidence,
    digest,
    origin_is_allowed,
)

MAX_SOURCE_BYTES: Final = 1_000_000
MAX_PLISTS: Final = 8
DISCOVERY_TIMEOUT_SECONDS: Final = 15.0
MAX_INTERPRETER_BYTES: Final = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, argv: tuple[str, ...], timeout: float, /) -> CommandResult: ...


def _run(argv: tuple[str, ...], timeout: float) -> CommandResult:
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False,
        env={"PATH": os.defpath, "LC_ALL": "C"},
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    agents_dir: Path = field(default_factory=lambda: Path.home() / "Library/LaunchAgents")
    domain: str = field(default_factory=lambda: f"gui/{os.getuid()}")
    identity: ServiceIdentity | None = None


@dataclass(slots=True)
class _DiscoveryError(Exception):
    state: State


def _read_source(path: Path) -> bytes:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_SOURCE_BYTES:
        raise _DiscoveryError(State.DISCOVERY_UNREADABLE)
    with path.open("rb") as handle:
        content = handle.read(MAX_SOURCE_BYTES + 1)
    if len(content) > MAX_SOURCE_BYTES:
        raise _DiscoveryError(State.DISCOVERY_UNREADABLE)
    return content


def _plist_fields(content: bytes, label: str) -> tuple[tuple[str, ...], str | None]:
    raw: JsonValue = plistlib.loads(content)
    match raw:
        case {"Label": str() as actual_label, "ProgramArguments": list() as arguments} if actual_label == label:
            args: list[str] = []
            for argument in arguments:
                if not isinstance(argument, str) or not argument or len(argument) > 4096:
                    raise _DiscoveryError(State.DISCOVERY_UNREADABLE)
                args.append(argument)
            if not 2 <= len(args) <= 64:
                raise _DiscoveryError(State.DISCOVERY_UNREADABLE)
            environment = raw.get("EnvironmentVariables")
            port = environment.get("PORT") if isinstance(environment, dict) else None
            return tuple(args), str(port) if type(port) in {str, int} else None
        case _:
            raise _DiscoveryError(State.DISCOVERY_UNREADABLE)


def _pid(result: CommandResult) -> int | None:
    if result.returncode != 0:
        if "Could not find service" in result.stderr:
            return None
        raise _DiscoveryError(State.DISCOVERY_BLOCKED)
    matches = re.findall(r"^\s*pid = ([0-9]+)\s*$", result.stdout, re.MULTILINE)
    if len(matches) != 1 or int(matches[0]) <= 0:
        raise _DiscoveryError(State.DISCOVERY_UNLOADED)
    return int(matches[0])


def _process_args(output: CommandResult) -> tuple[str, tuple[str, ...]]:
    fields = output.stdout.strip().split(maxsplit=5)
    if output.returncode != 0 or len(fields) != 6:
        raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
    args = tuple(shlex.split(fields[5]))
    if len(args) < 2:
        raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
    return " ".join(fields[:5]), args


def _binary_record(path: Path) -> tuple[str, int, int, int, int, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o002 or not before.st_mode & 0o111 or before.st_size > MAX_INTERPRETER_BYTES:
            raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
        hasher = hashlib.sha256()
        total = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            if total > MAX_INTERPRETER_BYTES:
                raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
            hasher.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise _DiscoveryError(State.DISCOVERY_CONFLICT)
    return str(path), after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode), after.st_size, hasher.hexdigest()


def _interpreter_identity(expected: Path, actual: Path) -> Sha256:
    if not expected.is_absolute() or not actual.is_absolute():
        raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
    expected_real, actual_real = expected.resolve(), actual.resolve()
    if expected_real == actual_real:
        return digest(json.dumps((_binary_record(expected_real),), separators=(",", ":")).encode())
    version_root = expected_real.parent.parent
    version = version_root.name
    helper = version_root / "Resources/Python.app/Contents/MacOS/Python"
    if (
        re.fullmatch(r"[0-9]+\.[0-9]+", version) is None
        or expected_real.parent.name != "bin"
        or expected_real.name != f"python{version}"
        or version_root.parent.name != "Versions"
        or version_root.parent.parent.name != "Python.framework"
        or helper.resolve() != helper
        or actual_real != helper
    ):
        raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
    launcher_record, helper_record = _binary_record(expected_real), _binary_record(helper)
    if launcher_record[1:3] != helper_record[1:3]:
        raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
    return digest(json.dumps((launcher_record, helper_record), separators=(",", ":")).encode())


def _target(script: Path, source: bytes, port: str | None, loaded: CommandResult) -> str:
    if script.name == "dashboard_moved.py":
        values: list[str] = []
        for node in ast.parse(source).body:
            match node:
                case ast.AnnAssign(target=ast.Name(id="ORIGIN"), value=ast.Constant(value=str() as value)):
                    values.append(value)
                case ast.Assign(targets=[ast.Name(id="ORIGIN")], value=ast.Constant(value=str() as value)):
                    values.append(value)
        if len(values) != 1 or not origin_is_allowed(values[0]):
            raise _DiscoveryError(State.DISCOVERY_UNREADABLE)
        return values[0].rstrip("/")
    if script.name == "rpms_service.py" and port is not None and port.isdigit() and 0 < int(port) < 65536:
        actual_ports = re.findall(r"^\s*PORT => ([0-9]+)\s*$", loaded.stdout, re.MULTILINE)
        if actual_ports != [port]:
            raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
        return f"https://127.0.0.1:{port}"
    raise _DiscoveryError(State.DISCOVERY_UNREADABLE)


@dataclass(frozen=True, slots=True)
class MacDiscovery:
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    run: CommandRunner = _run
    control: ProbeControl | None = None

    def _command(self, argv: tuple[str, ...], deadline: float) -> CommandResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (self.control is not None and self.control.expired):
            raise _DiscoveryError(State.DISCOVERY_TIMEOUT)
        return self.run(argv, remaining)

    def __call__(self) -> Discovery:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_SECONDS
        if self.control is not None:
            deadline = min(deadline, self.control.deadline)
            if self.control.expired:
                return Discovery(blocker=State.DISCOVERY_TIMEOUT)
        targets: list[TargetEvidence] = []
        try:
            if not self.config.agents_dir.is_dir():
                return Discovery(blocker=State.DISCOVERY_UNREADABLE)
            paths = sorted(self.config.agents_dir.glob("com.bluemango.dashboard*.plist"))
            if len(paths) > MAX_PLISTS:
                return Discovery(blocker=State.DISCOVERY_BLOCKED)
            for path in paths:
                loaded = self._command(("/bin/launchctl", "print", f"{self.config.domain}/{path.stem}"), deadline)
                pid = _pid(loaded)
                if pid is None:
                    continue
                plist_bytes = _read_source(path)
                expected_args, port = _plist_fields(plist_bytes, path.stem)
                scripts = [Path(arg) for arg in expected_args if arg.endswith(".py")]
                if len(scripts) != 1 or not scripts[0].is_absolute():
                    raise _DiscoveryError(State.DISCOVERY_UNREADABLE)
                script = scripts[0]
                source_bytes = _read_source(script)
                origin = _target(script, source_bytes, port, loaded)
                process = self._command(("/bin/ps", "-p", str(pid), "-ww", "-o", "lstart=", "-o", "command="), deadline)
                started_at, actual_args = _process_args(process)
                if actual_args[1:] != expected_args[1:]:
                    raise _DiscoveryError(State.DISCOVERY_PROCESS_MISMATCH)
                interpreter = _interpreter_identity(Path(expected_args[0]), Path(actual_args[0]))
                reloaded = self._command(("/bin/launchctl", "print", f"{self.config.domain}/{path.stem}"), deadline)
                reprocess = self._command(("/bin/ps", "-p", str(pid), "-ww", "-o", "lstart=", "-o", "command="), deadline)
                if _pid(reloaded) != pid or _process_args(reprocess) != (started_at, actual_args) or _read_source(path) != plist_bytes or _read_source(script) != source_bytes or _interpreter_identity(Path(expected_args[0]), Path(actual_args[0])) != interpreter:
                    raise _DiscoveryError(State.DISCOVERY_CONFLICT)
                fingerprint = digest(json.dumps({"plist": digest(plist_bytes), "source": digest(source_bytes), "pid": pid, "started_at": started_at, "argv": actual_args, "interpreter": interpreter, "origin": origin}, sort_keys=True, separators=(",", ":")).encode())
                if time.monotonic() >= deadline:
                    raise _DiscoveryError(State.DISCOVERY_TIMEOUT)
                targets.append(TargetEvidence(origin, fingerprint, time.time(), self.config.identity))
        except _DiscoveryError as failure:
            return Discovery(blocker=failure.state)
        except subprocess.TimeoutExpired:
            return Discovery(blocker=State.DISCOVERY_TIMEOUT)
        except (OSError, ValueError, SyntaxError, plistlib.InvalidFileException):
            return Discovery(blocker=State.DISCOVERY_UNREADABLE)
        return Discovery(tuple(targets))
