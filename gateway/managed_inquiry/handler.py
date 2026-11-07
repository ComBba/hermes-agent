from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import assert_never

from .models import ProbeInputError, ProbeReport
from .renderer import render, render_probe
from .source import SQLiteInquirySource
from .task_models import (
    BindingRole,
    CompletionDraft,
    CompletionState,
    InquiryRequest,
    PreparedCompletion,
    RequestControl,
    TaskKind,
    TaskSelection,
    WorkClaim,
)
from .task_state import PendingStore


@dataclass(frozen=True, slots=True)
class InquiryServices:
    state: PendingStore
    probe: Callable[[RequestControl], ProbeReport]
    source: SQLiteInquirySource


def _valid_owner(request: InquiryRequest) -> bool:
    execution, ingress, scope = request.execution, request.ingress, request.scope
    if execution is None or execution.profile_id != scope.profile_id or execution.session_id != scope.session_id:
        return False
    if ingress is None:
        return True
    return bool(ingress.profile_id == scope.profile_id and ingress.session_id == scope.session_id
                and ingress.platform == scope.platform and ingress.actor_id == scope.actor_id
                and ingress.message_id == request.message_id and ingress.admission_id == execution.admission_id
                and ingress.admission_generation == execution.admission_generation
                and ingress.authority_epoch == execution.authority_epoch and ingress.principal_id == execution.principal_id)


def _action(text: str) -> str | None:
    parts = text.strip().split(maxsplit=2)
    if not parts or parts[0] != "/inquiry":
        return None
    return parts[1] if len(parts) > 1 else "invalid"


def handle(request: InquiryRequest, services: InquiryServices) -> PreparedCompletion | None:
    if request.role == BindingRole.NONE or request.internal:
        return None
    if not _valid_owner(request):
        if request.role == BindingRole.COORDINATOR:
            return None
        raise ProbeInputError("request_owner")
    cached = services.state.cached(request)
    if cached is not None:
        return cached
    pending = services.state.pending(request.scope)
    action = _action(request.text)
    if action == "advice" and pending is None:
        return None
    selection: TaskSelection
    immediate: CompletionState | None = None
    if pending is not None:
        source = replace(pending.source, request_sha256=request.identity_sha256) if pending.source is not None else None
        selection = TaskSelection(pending.kind, source, action == "reset")
        if action == "advice":
            immediate = CompletionState.PENDING_TASK_BLOCKED
    elif action in {"status", "investigate", "cancel", "reset", "invalid"}:
        selection = TaskSelection(TaskKind.INVESTIGATION if action == "investigate" else TaskKind.STATUS, reset=action == "reset")
    elif action is not None:
        selection = TaskSelection(TaskKind.STATUS)
        immediate = CompletionState.INPUT_BLOCKED
    elif request.role == BindingRole.INQUIRY:
        selection = TaskSelection(TaskKind.STATUS)
    else:
        if request.control.expired:
            return None
        source = services.source.find(request, services.state.lookup_reply(request))
        if source is None:
            return None
        selection = TaskSelection(TaskKind.INVESTIGATION, source)
    if action in {"cancel", "reset"}:
        immediate = CompletionState.CANCELLED
    if not request.text.strip() or len(request.text) > 16000:
        immediate = CompletionState.INPUT_BLOCKED
    if selection.kind == TaskKind.INVESTIGATION and selection.source is None and immediate is None:
        selected_source = services.source.find(request, services.state.lookup_reply(request))
        selection = replace(selection, source=selected_source)
    claim = services.state.begin(request, selection)
    match claim:
        case None:
            return None
        case PreparedCompletion():
            return claim
        case WorkClaim():
            pass
        case unreachable:
            assert_never(unreachable)
    if claim.token is None:
        while not request.control.expired:
            completed = services.state.cached(request)
            if completed is not None:
                return completed
            request.control.cancelled.wait(min(0.01, max(0.0, request.control.deadline - time.monotonic())))
        return services.state.abort(request)
    if immediate is not None:
        return services.state.finish(claim, render(immediate, ""))
    try:
        draft: CompletionDraft
        if claim.task.kind == TaskKind.INVESTIGATION:
            source = claim.task.source
            if source is None:
                draft = render(CompletionState.EVIDENCE_UNLINKED, "")
            else:
                inspection = services.source.inspect(request, source)
                draft = render_probe(services.probe(request.control), inspection) if inspection.inspected else render(CompletionState.INSPECTION_BLOCKED, inspection.evidence_sha256)
        else:
            draft = render_probe(services.probe(request.control))
        return services.state.finish(claim, draft)
    except (OSError, sqlite3.Error, ValueError, RuntimeError, TimeoutError):
        return services.state.finish(claim, render(CompletionState.HANDLER_ERROR, ""))


def timeout(request: InquiryRequest, services: InquiryServices) -> PreparedCompletion | None:
    return services.state.abort(request)


def failed(request: InquiryRequest, services: InquiryServices) -> PreparedCompletion | None:
    return services.state.abort(request, CompletionState.HANDLER_ERROR)
