from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final, NewType, Protocol, TypeAlias, override
from urllib.parse import urlsplit

SCHEMA_VERSION: Final = 1
Sha256 = NewType("Sha256", str)
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class ProbeControl(Protocol):
    @property
    def deadline(self) -> float: ...
    @property
    def expired(self) -> bool: ...


class State(StrEnum):
    HEALTHY = "healthy"
    AUTH_BLOCKED = "auth_blocked"
    TRANSPORT_BLOCKED = "transport_blocked"
    TLS_BLOCKED = "tls_blocked"
    RESPONSE_TOO_LARGE = "response_too_large"
    TARGET_MOVED = "target_moved"
    UNEXPECTED_HTML = "unexpected_html"
    MIME_INVALID = "mime_invalid"
    SCHEMA_INVALID = "schema_invalid"
    UNHEALTHY = "unhealthy"
    IDENTITY_UNVERIFIED = "identity_unverified"
    IDENTITY_MISMATCH = "identity_mismatch"
    FRESHNESS_UNVERIFIED = "freshness_unverified"
    DATA_STALE = "data_stale"
    DISCOVERY_BLOCKED = "discovery_blocked"
    DISCOVERY_TIMEOUT = "discovery_timeout"
    DISCOVERY_UNREADABLE = "discovery_unreadable"
    DISCOVERY_CONFLICT = "discovery_conflict"
    DISCOVERY_UNLOADED = "discovery_unloaded"
    DISCOVERY_PROCESS_MISMATCH = "discovery_process_mismatch"
    DISCOVERY_STALE = "discovery_stale"
    REDIRECT_UNAPPROVED = "redirect_unapproved"
    REDISCOVERY_EXHAUSTED = "rediscovery_exhausted"


@dataclass(slots=True)
class ProbeInputError(ValueError):
    field: str

    @override
    def __str__(self) -> str:
        return f"invalid probe input: {self.field}"


def digest(content: bytes) -> Sha256:
    return Sha256(hashlib.sha256(content).hexdigest())


def is_loopback(host: str | None) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def origin_is_allowed(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path in {"", "/"}
        and (port is None or 0 < port < 65536)
        and (parsed.scheme == "https" or (parsed.scheme == "http" and is_loopback(parsed.hostname)))
    )


@dataclass(frozen=True, slots=True)
class ServiceIdentity:
    field: str
    value: str

    def __post_init__(self) -> None:
        if not self.field or not self.value:
            raise ProbeInputError("identity")


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    origin: str
    source_sha256: Sha256
    observed_at: float
    identity: ServiceIdentity | None = None

    def __post_init__(self) -> None:
        if not origin_is_allowed(self.origin):
            raise ProbeInputError("origin")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ProbeInputError("source_sha256")
        if not math.isfinite(self.observed_at) or self.observed_at < 0:
            raise ProbeInputError("observed_at")


@dataclass(frozen=True, slots=True)
class Discovery:
    targets: tuple[TargetEvidence, ...] = ()
    blocker: State | None = None


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    content_type: str
    body: bytes


@dataclass(frozen=True, slots=True)
class TransportFailure:
    state: State


@dataclass(frozen=True, slots=True)
class ProbePolicy:
    data_stale_sec: float = 900.0
    source_max_age_sec: float = 30.0
    now: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not math.isfinite(self.data_stale_sec) or self.data_stale_sec < 0:
            raise ProbeInputError("data_stale_sec")
        if not math.isfinite(self.source_max_age_sec) or self.source_max_age_sec < 0:
            raise ProbeInputError("source_max_age_sec")


@dataclass(frozen=True, slots=True)
class HealthObservation:
    state: State
    http_status: int
    response_sha256: Sha256
    health_ok: bool | None = None
    identity_verified: bool = False
    data_age_sec: float | None = None


@dataclass(frozen=True, slots=True)
class ProbeReport:
    state: State
    target: TargetEvidence | None
    observations: tuple[HealthObservation, ...]
    discovery_count: int
    schema_version: int = SCHEMA_VERSION


def report_to_json(report: ProbeReport) -> str:
    return json.dumps(asdict(report), sort_keys=True, separators=(",", ":"), allow_nan=False)
