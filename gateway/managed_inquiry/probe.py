from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Final, TypeAlias, assert_never

from .models import (
    Discovery,
    HealthObservation,
    HttpResponse,
    JsonValue,
    ProbeControl,
    ProbePolicy,
    ProbeReport,
    State,
    TargetEvidence,
    TransportFailure,
    digest,
)
from .transport import fetch_health

DiscoveryReader: TypeAlias = Callable[[], Discovery]
Fetcher: TypeAlias = Callable[[TargetEvidence], HttpResponse | TransportFailure]
DEFAULT_POLICY: Final = ProbePolicy()


class _JsonInvalid(ValueError):
    pass


def _reject_constant(_value: str) -> None:
    raise _JsonInvalid()


def _unique_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _JsonInvalid()
        result[key] = value
    return result


def classify(response: HttpResponse, target: TargetEvidence, policy: ProbePolicy) -> HealthObservation:
    observation = HealthObservation(State.SCHEMA_INVALID, response.status, digest(response.body))
    if response.status in {401, 403}:
        return replace(observation, state=State.AUTH_BLOCKED)
    if 300 <= response.status < 400:
        return replace(observation, state=State.REDIRECT_UNAPPROVED)
    mime = response.content_type.split(";", 1)[0].strip().lower()
    try:
        text = response.body.decode("utf-8")
    except UnicodeDecodeError:
        return observation
    if mime in {"text/html", "application/xhtml+xml"} or text.lstrip().lower().startswith(("<!doctype", "<html")):
        title = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        moved = title is not None and any(term in title.group(1).casefold() for term in ("moved", "주소 변경"))
        return replace(observation, state=State.TARGET_MOVED if moved else State.UNEXPECTED_HTML)
    if mime != "application/json":
        return replace(observation, state=State.MIME_INVALID)
    try:
        payload: JsonValue = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError):
        return observation
    match payload:
        case dict() as fields:
            health_ok = fields.get("ok")
        case str() | int() | float() | bool() | list() | None:
            return observation
        case unreachable:
            assert_never(unreachable)
    if type(health_ok) is not bool:
        return observation
    observation = replace(observation, health_ok=health_ok)
    if target.identity is None:
        return replace(observation, state=State.IDENTITY_UNVERIFIED)
    if fields.get(target.identity.field) != target.identity.value:
        return replace(observation, state=State.IDENTITY_MISMATCH)
    observation = replace(observation, identity_verified=True)
    if not health_ok or response.status != 200:
        return replace(observation, state=State.UNHEALTHY)
    age = fields.get("data_age_sec")
    match age:
        case bool() | str() | list() | dict() | None:
            return replace(observation, state=State.FRESHNESS_UNVERIFIED)
        case int() | float():
            pass
        case unreachable:
            assert_never(unreachable)
    try:
        numeric_age = float(age)
    except OverflowError:
        return replace(observation, state=State.FRESHNESS_UNVERIFIED)
    if not math.isfinite(numeric_age) or numeric_age < 0:
        return replace(observation, state=State.FRESHNESS_UNVERIFIED)
    observation = replace(observation, data_age_sec=numeric_age)
    if numeric_age > policy.data_stale_sec:
        return replace(observation, state=State.DATA_STALE)
    return replace(observation, state=State.HEALTHY)


def _resolved(discovery: Discovery, policy: ProbePolicy) -> TargetEvidence | State:
    if discovery.blocker is not None:
        return discovery.blocker
    if not discovery.targets:
        return State.DISCOVERY_UNLOADED
    now = policy.now()
    if not math.isfinite(now) or any(not -5 <= now - target.observed_at <= policy.source_max_age_sec for target in discovery.targets):
        return State.DISCOVERY_STALE
    if len({target.origin.rstrip("/") for target in discovery.targets}) != 1 or len({target.identity for target in discovery.targets}) != 1:
        return State.DISCOVERY_CONFLICT
    first = discovery.targets[0]
    if len(discovery.targets) == 1:
        return first
    fingerprint = digest(json.dumps(sorted(target.source_sha256 for target in discovery.targets), separators=(",", ":")).encode())
    return replace(first, source_sha256=fingerprint, observed_at=min(target.observed_at for target in discovery.targets))


def probe(discover: DiscoveryReader, fetch: Fetcher = fetch_health, policy: ProbePolicy = DEFAULT_POLICY, *, control: ProbeControl | None = None) -> ProbeReport:
    if control is not None and control.expired:
        return ProbeReport(State.DISCOVERY_TIMEOUT, None, (), 0)
    current = _resolved(discover(), policy)
    match current:
        case State() as failure:
            return ProbeReport(failure, None, (), 1)
        case TargetEvidence() as target:
            pass
        case unreachable:
            assert_never(unreachable)
    observations: tuple[HealthObservation, ...] = ()
    for attempt in range(2):
        if control is not None and control.expired:
            return ProbeReport(State.DISCOVERY_TIMEOUT, target, observations, attempt + 1)
        fetched = fetch_health(target, control=control) if fetch is fetch_health else fetch(target)
        match fetched:
            case TransportFailure(state=state):
                return ProbeReport(state, target, observations, attempt + 1)
            case HttpResponse():
                observation = classify(fetched, target, policy)
            case unreachable:
                assert_never(unreachable)
        observations += (observation,)
        if observation.state not in {
            State.TARGET_MOVED, State.UNEXPECTED_HTML, State.MIME_INVALID,
            State.SCHEMA_INVALID, State.REDIRECT_UNAPPROVED,
        }:
            return ProbeReport(observation.state, target, observations, attempt + 1)
        if attempt == 1:
            return ProbeReport(State.REDISCOVERY_EXHAUSTED, target, observations, 2)
        if control is not None and control.expired:
            return ProbeReport(State.DISCOVERY_TIMEOUT, target, observations, 1)
        refreshed = _resolved(discover(), policy)
        match refreshed:
            case State() as failure:
                return ProbeReport(failure, target, observations, 2)
            case TargetEvidence() as replacement:
                if replacement.source_sha256 == target.source_sha256:
                    return ProbeReport(State.REDISCOVERY_EXHAUSTED, target, observations, 2)
                target = replacement
            case unreachable:
                assert_never(unreachable)
    raise AssertionError("bounded probe exhausted without a result")
