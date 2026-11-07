from __future__ import annotations

import http.client
import ssl
import time
from contextlib import closing
from email.message import Message
from typing import IO, Final, override
from urllib import error, request
from urllib.parse import urlsplit

from .models import (
    HttpResponse,
    ProbeControl,
    State,
    TargetEvidence,
    TransportFailure,
    is_loopback,
)

MAX_RESPONSE_BYTES: Final = 200_000
FETCH_TIMEOUT_SECONDS: Final = 15


class _RejectRedirect(request.HTTPRedirectHandler):
    @override
    def redirect_request(self, req: request.Request, fp: IO[bytes], code: int, msg: str, headers: Message, newurl: str) -> None:
        return None


def fetch_health(target: TargetEvidence, *, control: ProbeControl | None = None) -> HttpResponse | TransportFailure:
    deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    if control is not None:
        deadline = min(deadline, control.deadline)
    if control is not None and control.expired:
        return TransportFailure(State.DISCOVERY_TIMEOUT)
    context = ssl.create_default_context()
    if is_loopback(urlsplit(target.origin).hostname):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    opener = request.build_opener(
        request.ProxyHandler({}),
        request.HTTPSHandler(context=context),
        _RejectRedirect(),
    )
    outgoing = request.Request(
        target.origin.rstrip("/") + "/healthz",
        method="GET",
        headers={"User-Agent": "managed-inquiry-status/1", "Accept": "application/json"},
    )
    try:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return TransportFailure(State.DISCOVERY_TIMEOUT)
            incoming = opener.open(outgoing, timeout=remaining)
        except error.HTTPError as response:
            incoming = response
        with closing(incoming):
            body = bytearray()
            while len(body) <= MAX_RESPONSE_BYTES:
                if time.monotonic() >= deadline or (control is not None and control.expired):
                    return TransportFailure(State.DISCOVERY_TIMEOUT)
                chunk = incoming.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                return TransportFailure(State.RESPONSE_TOO_LARGE)
            return HttpResponse(incoming.code, incoming.headers.get("Content-Type", ""), bytes(body))
    except (ssl.SSLError, ssl.CertificateError):
        return TransportFailure(State.TLS_BLOCKED)
    except error.URLError as failure:
        if isinstance(failure.reason, ssl.SSLError):
            return TransportFailure(State.TLS_BLOCKED)
        return TransportFailure(State.TRANSPORT_BLOCKED)
    except (OSError, TimeoutError, http.client.HTTPException):
        return TransportFailure(State.TRANSPORT_BLOCKED)
