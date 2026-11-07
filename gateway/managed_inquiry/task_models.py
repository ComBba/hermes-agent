from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from .ingress_contract import ExecutionView, IngressView
from .models import ProbeInputError, digest


def _request_identity(scope: TaskScope, execution: ExecutionView, binding: str | None) -> str:
    fields = (scope.key, execution.admission_id, execution.admission_generation,
              execution.authority_epoch, execution.principal_id, execution.request_id, binding)
    return digest(json.dumps(fields, separators=(",", ":")).encode())


class BindingRole(StrEnum):
    NONE = "none"
    INQUIRY = "inquiry"
    COORDINATOR = "coordinator"


class TaskKind(StrEnum):
    STATUS = "status"
    INVESTIGATION = "investigation"


class CompletionState(StrEnum):
    HEALTH_VERIFIED = "health_verified"
    AUTH_BLOCKED = "auth_blocked"
    STATUS_BLOCKED = "status_blocked"
    INSPECTED = "inspected"
    EVIDENCE_UNLINKED = "evidence_unlinked"
    INSPECTION_BLOCKED = "inspection_blocked"
    INPUT_BLOCKED = "input_blocked"
    PENDING_TASK_BLOCKED = "pending_task_blocked"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    HANDLER_ERROR = "handler_error"


class Finding(StrEnum):
    HEALTH_PARSE_FAILED = "health_parse_failed"
    MIGRATION_HTML = "migration_html"
    RECOVERY_NOT_OBSERVED = "recovery_not_observed"


@dataclass(frozen=True, slots=True)
class TaskScope:
    profile_id: str
    platform: str
    channel_id: str
    thread_id: str
    actor_id: str
    session_id: str
    runtime_profile: str = ""
    source_profile: str | None = None

    @property
    def key(self) -> str:
        return digest(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode())

    @property
    def group_key(self) -> str:
        fields = (self.profile_id, self.runtime_profile, self.source_profile, self.platform, self.channel_id, self.thread_id, self.actor_id)
        return digest(json.dumps(fields, separators=(",", ":")).encode())


@dataclass(frozen=True, slots=True)
class RequestControl:
    deadline: float = field(default_factory=lambda: time.monotonic() + 90.0)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        if not math.isfinite(self.deadline):
            raise ProbeInputError("deadline")

    @property
    def expired(self) -> bool:
        return self.cancelled.is_set() or time.monotonic() >= self.deadline

    def cancel(self) -> None:
        self.cancelled.set()


@dataclass(frozen=True, slots=True)
class InquiryRequest:
    scope: TaskScope
    role: BindingRole
    text: str
    message_id: str
    execution: ExecutionView | None
    ingress: IngressView | None = None
    reply_to_message_id: str | None = None
    internal: bool = False
    control: RequestControl = field(default_factory=RequestControl)

    @property
    def identity_sha256(self) -> str:
        execution = self.execution
        if execution is None:
            raise ProbeInputError("execution")
        return _request_identity(self.scope, execution, self.ingress.binding_sha256 if self.ingress is not None else None)


@dataclass(frozen=True, slots=True)
class SourceAdmission:
    admission_id: str
    admission_generation: int
    authority_epoch: int
    payload_sha256: str
    principal_id: str


@dataclass(frozen=True, slots=True)
class SourceMessage:
    row_id: int
    role: str
    content_sha256: str
    tool_call_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SourceLink:
    scope_key: str
    request_sha256: str
    source_session_id: str
    source_admissions: tuple[SourceAdmission, ...]
    messages: tuple[SourceMessage, ...]
    image_sha256: str | None

    @property
    def evidence_sha256(self) -> str:
        return digest(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True, slots=True)
class Inspection:
    inspected: bool
    evidence_sha256: str
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True, slots=True)
class CompletionDraft:
    state: CompletionState
    text: str
    retains_pending: bool
    evidence_sha256: str
    detail_code: str = ""


@dataclass(frozen=True, slots=True)
class PendingTask:
    task_id: str
    kind: TaskKind
    pending_reset_epoch: int
    source: SourceLink | None


@dataclass(frozen=True, slots=True)
class TaskSelection:
    kind: TaskKind
    source: SourceLink | None = None
    reset: bool = False


@dataclass(frozen=True, slots=True)
class WorkClaim:
    request: InquiryRequest
    task: PendingTask
    token: str | None


@dataclass(frozen=True, slots=True)
class DeliveryReference:
    admission_id: str
    request_id: str
    source_session_id: str


@dataclass(frozen=True, slots=True)
class StoredExecution:
    profile_id: str
    session_id: str
    admission_id: str
    admission_generation: int
    authority_epoch: int
    principal_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class DeliveryBinding:
    scope: TaskScope
    execution: StoredExecution
    message_id: str
    ingress_binding_sha256: str | None
    connector: str
    reply_to_message_id: str | None

    @property
    def identity_sha256(self) -> str:
        return _request_identity(self.scope, self.execution, self.ingress_binding_sha256)


@dataclass(frozen=True, slots=True)
class PreparedCompletion:
    completion_id: str
    scope_key: str
    task_id: str
    kind: TaskKind
    pending_reset_epoch: int
    admission_id: str
    admission_generation: int
    authority_epoch: int
    request_sha256: str
    state: CompletionState
    text: str
    retains_pending: bool
    evidence_sha256: str
    receipt_sha256: str
    detail_code: str = ""


@dataclass(frozen=True, slots=True)
class DeliverySnapshot:
    completion: PreparedCompletion
    binding: DeliveryBinding
    source: SourceLink | None
