from __future__ import annotations

import base64
import binascii
import http.client
import json
import math
import os
import selectors
import ssl
import subprocess
import sys
import time
from contextlib import closing
from email.message import Message
from pathlib import Path
from typing import IO, Final, NoReturn, override
from urllib import error, request
from urllib.parse import urlsplit

from .models import (
    HttpResponse,
    JsonValue,
    ProbeControl,
    State,
    TargetEvidence,
    TransportFailure,
    is_loopback,
    origin_is_allowed,
)

MAX_RESPONSE_BYTES: Final = 200_000
FETCH_TIMEOUT_SECONDS: Final = 15
MAX_CONTENT_TYPE_BYTES: Final = 2048
_MAX_STDOUT: Final = 300_000
_MAX_STDERR: Final = 4096
_WORKER_BOOTSTRAP: Final = (
    "import importlib,sys;from pathlib import Path;"
    "sys.path.insert(0,sys.argv[1]);m=importlib.import_module(sys.argv[2]);"
    "assert Path(m.__file__).resolve()==Path(sys.argv[3]);m._worker_main()"
)


class _RejectRedirect(request.HTTPRedirectHandler):
    @override
    def redirect_request(
        self,
        req: request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> None:
        return None


def _fetch_response(origin: str, timeout: float) -> HttpResponse | TransportFailure:
    context = ssl.create_default_context()
    if is_loopback(urlsplit(origin).hostname):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    opener = request.build_opener(
        request.ProxyHandler({}),
        request.HTTPSHandler(context=context),
        _RejectRedirect(),
    )
    outgoing = request.Request(
        origin.rstrip("/") + "/healthz",
        method="GET",
        headers={
            "User-Agent": "managed-inquiry-status/1",
            "Accept": "application/json",
        },
    )
    try:
        try:
            incoming = opener.open(outgoing, timeout=timeout)
        except error.HTTPError as response:
            incoming = response
        with closing(incoming):
            content_type = incoming.headers.get("Content-Type", "")
            if len(content_type.encode("utf-8")) > MAX_CONTENT_TYPE_BYTES:
                return TransportFailure(State.RESPONSE_TOO_LARGE)
            body = bytearray()
            while len(body) <= MAX_RESPONSE_BYTES:
                chunk = incoming.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                return TransportFailure(State.RESPONSE_TOO_LARGE)
            return HttpResponse(incoming.code, content_type, bytes(body))
    except (ssl.SSLError, ssl.CertificateError):
        return TransportFailure(State.TLS_BLOCKED)
    except error.URLError as failure:
        return TransportFailure(
            State.TLS_BLOCKED
            if isinstance(failure.reason, ssl.SSLError)
            else State.TRANSPORT_BLOCKED
        )
    except (OSError, TimeoutError, http.client.HTTPException):
        return TransportFailure(State.TRANSPORT_BLOCKED)


def _unique_pairs(rows: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in rows:
        if key in value:
            raise ValueError("duplicate frame field")
        value[key] = item
    return value


def _nonfinite(_value: str) -> NoReturn:
    raise ValueError("nonfinite frame value")


def _frame(data: bytes, maximum: int) -> dict[str, JsonValue]:
    if len(data) > maximum or not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise ValueError("invalid frame boundary")
    value: JsonValue = json.loads(
        data[:-1], object_pairs_hook=_unique_pairs, parse_constant=_nonfinite
    )
    if not isinstance(value, dict):
        raise ValueError("invalid frame object")
    return value


def _decode_response(data: bytes) -> HttpResponse | TransportFailure:
    value = _frame(data, _MAX_STDOUT)
    if value.get("kind") == "failure" and set(value) == {"kind", "state"}:
        state = value["state"]
        if not isinstance(state, str) or state not in {
            State.TRANSPORT_BLOCKED,
            State.TLS_BLOCKED,
            State.RESPONSE_TOO_LARGE,
            State.DISCOVERY_TIMEOUT,
        }:
            raise ValueError("invalid failure state")
        return TransportFailure(State(state))
    if value.get("kind") != "response" or set(value) != {
        "kind",
        "status",
        "content_type",
        "body",
    }:
        raise ValueError("invalid response fields")
    status, content_type, encoded = (
        value["status"],
        value["content_type"],
        value["body"],
    )
    if (
        type(status) is not int
        or not 100 <= status <= 599
        or not isinstance(content_type, str)
        or len(content_type.encode("utf-8")) > MAX_CONTENT_TYPE_BYTES
        or not isinstance(encoded, str)
    ):
        raise ValueError("invalid response values")
    body = base64.b64decode(encoded, validate=True)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("invalid response size")
    return HttpResponse(status, content_type, body)


def _worker_main() -> None:
    try:
        value = _frame(sys.stdin.buffer.read(8193), 8192)
        origin, timeout = value.get("origin"), value.get("timeout")
        if (
            set(value) != {"origin", "timeout"}
            or not isinstance(origin, str)
            or len(origin) > 4096
            or not origin_is_allowed(origin)
            or type(timeout) not in {int, float}
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= FETCH_TIMEOUT_SECONDS
        ):
            raise ValueError("invalid request frame")
        result = _fetch_response(origin, float(timeout))
        if isinstance(result, TransportFailure):
            output = {"kind": "failure", "state": result.state.value}
        else:
            output = {
                "kind": "response",
                "status": result.status,
                "content_type": result.content_type,
                "body": base64.b64encode(result.body).decode("ascii"),
            }
        data = (
            json.dumps(
                output, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            ).encode()
            + b"\n"
        )
        if len(data) > _MAX_STDOUT:
            raise ValueError("oversize response frame")
        _ = sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except (ValueError, TypeError, OSError, RecursionError):
        raise SystemExit(2) from None


def _worker_environment() -> dict[str, str]:
    environment = {"PATH": os.defpath}
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


def _reap_worker(worker: subprocess.Popen[bytes], deadline: float) -> bool:
    try:
        if worker.poll() is None:
            worker.terminate()
            try:
                _ = worker.wait(
                    timeout=max(0, min(0.1, (deadline - time.monotonic()) / 2))
                )
            except subprocess.TimeoutExpired:
                worker.kill()
                _ = worker.wait(timeout=max(0, deadline - time.monotonic()))
        else:
            _ = worker.wait(timeout=0)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            if stream is not None:
                stream.close()


def fetch_health(
    target: TargetEvidence, *, control: ProbeControl | None = None
) -> HttpResponse | TransportFailure:
    deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    if control is not None:
        deadline = min(deadline, control.deadline)
    if (control is not None and control.expired) or time.monotonic() >= deadline:
        return TransportFailure(State.DISCOVERY_TIMEOUT)
    if __name__ not in {
        "runtime_payload.managed_inquiry.transport",
        "gateway.managed_inquiry.transport",
    }:
        return TransportFailure(State.TRANSPORT_BLOCKED)
    work_deadline = deadline - min(0.25, max(0, deadline - time.monotonic()) * 0.2)
    source = Path(__file__).resolve()
    worker: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    response: HttpResponse | TransportFailure = TransportFailure(
        State.TRANSPORT_BLOCKED
    )
    try:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-u",
                "-c",
                _WORKER_BOOTSTRAP,
                str(source.parents[2]),
                __name__,
                str(source),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_worker_environment(),
            shell=False,
            start_new_session=True,
            bufsize=0,
        )
        assert (
            worker.stdin is not None
            and worker.stdout is not None
            and worker.stderr is not None
        )
        pending = memoryview(
            json.dumps(
                {
                    "origin": target.origin,
                    "timeout": max(
                        0.001, min(FETCH_TIMEOUT_SECONDS, deadline - time.monotonic())
                    ),
                },
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        if len(pending) > 8192:
            raise ValueError("request frame too large")
        for stream, events, label in (
            (worker.stdin, selectors.EVENT_WRITE, "in"),
            (worker.stdout, selectors.EVENT_READ, "out"),
            (worker.stderr, selectors.EVENT_READ, "err"),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, events, label)
        output = bytearray()
        errors = bytearray()
        while selector.get_map() or worker.poll() is None:
            if time.monotonic() >= work_deadline or (
                control is not None and control.expired
            ):
                response = TransportFailure(State.DISCOVERY_TIMEOUT)
                break
            for key, _ in selector.select(
                min(0.01, max(0, work_deadline - time.monotonic()))
            ):
                if key.data == "in":
                    written = os.write(key.fd, pending[:4096])
                    pending = pending[written:]
                    if not pending:
                        selector.unregister(worker.stdin)
                        worker.stdin.close()
                    continue
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer, limit = (
                    (output, _MAX_STDOUT)
                    if key.data == "out"
                    else (errors, _MAX_STDERR)
                )
                if len(buffer) + len(chunk) > limit:
                    raise ValueError("worker output too large")
                buffer.extend(chunk)
                if errors:
                    raise ValueError("worker stderr")
        else:
            if time.monotonic() >= deadline or (
                control is not None and control.expired
            ):
                response = TransportFailure(State.DISCOVERY_TIMEOUT)
            elif worker.returncode == 0:
                response = _decode_response(bytes(output))
    except (
        OSError,
        ValueError,
        TypeError,
        RecursionError,
        binascii.Error,
        subprocess.SubprocessError,
    ):
        response = TransportFailure(State.TRANSPORT_BLOCKED)
    finally:
        selector.close()
        if worker is not None and not _reap_worker(worker, deadline):
            response = TransportFailure(State.TRANSPORT_BLOCKED)
    return response
