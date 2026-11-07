from __future__ import annotations

import json
from dataclasses import asdict

from .models import JsonValue, ProbeInputError
from .task_models import (
    CompletionState,
    DeliveryBinding,
    InquiryRequest,
    PreparedCompletion,
    SourceAdmission,
    SourceLink,
    SourceMessage,
    StoredExecution,
    TaskKind,
    TaskScope,
)


def encode_link(link: SourceLink | None) -> str | None:
    return json.dumps(asdict(link), sort_keys=True, separators=(",", ":")) if link is not None else None


def fields(encoded: str) -> dict[str, JsonValue]:
    value: JsonValue = json.loads(encoded)
    if not isinstance(value, dict):
        raise ProbeInputError("stored_record")
    return value


def text(record: dict[str, JsonValue], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value) > 8192:
        raise ProbeInputError(key)
    return value


def number(record: dict[str, JsonValue], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise ProbeInputError(key)
    return value


def decode_link(encoded: str | None) -> SourceLink | None:
    if encoded is None:
        return None
    record = fields(encoded)
    admissions = record.get("source_admissions")
    messages = record.get("messages")
    if not isinstance(admissions, list) or not isinstance(messages, list) or len(admissions) > 256 or len(messages) > 128:
        raise ProbeInputError("source_records")
    admitted: list[SourceAdmission] = []
    observed: list[SourceMessage] = []
    for row in admissions:
        if not isinstance(row, dict):
            raise ProbeInputError("source_admission")
        admitted.append(SourceAdmission(text(row, "admission_id"), number(row, "admission_generation"), number(row, "authority_epoch"), text(row, "payload_sha256"), text(row, "principal_id")))
    for row in messages:
        if not isinstance(row, dict):
            raise ProbeInputError("source_message")
        tool_hash = row.get("tool_call_sha256")
        if tool_hash is not None and not isinstance(tool_hash, str):
            raise ProbeInputError("tool_call_sha256")
        observed.append(SourceMessage(number(row, "row_id"), text(row, "role"), text(row, "content_sha256"), tool_hash))
    image_hash = record.get("image_sha256")
    if image_hash is not None and not isinstance(image_hash, str):
        raise ProbeInputError("image_sha256")
    return SourceLink(text(record, "scope_key"), text(record, "request_sha256"), text(record, "source_session_id"), tuple(admitted), tuple(observed), image_hash)


def encode_completion(completion: PreparedCompletion) -> str:
    return json.dumps(asdict(completion), sort_keys=True, separators=(",", ":"))


def decode_completion(encoded: str) -> PreparedCompletion:
    record = fields(encoded)
    retains_pending = record.get("retains_pending")
    if type(retains_pending) is not bool:
        raise ProbeInputError("retains_pending")
    return PreparedCompletion(
        text(record, "completion_id"), text(record, "scope_key"), text(record, "task_id"),
        TaskKind(text(record, "kind")), number(record, "pending_reset_epoch"),
        text(record, "admission_id"), number(record, "admission_generation"), number(record, "authority_epoch"),
        text(record, "request_sha256"), CompletionState(text(record, "state")), text(record, "text"),
        retains_pending, text(record, "evidence_sha256"), text(record, "receipt_sha256"),
        text(record, "detail_code"),
    )


def delivery_binding(request: InquiryRequest) -> DeliveryBinding:
    execution, ingress = request.execution, request.ingress
    if execution is None:
        raise ProbeInputError("execution")
    stored = StoredExecution(execution.profile_id, execution.session_id, execution.admission_id,
                             execution.admission_generation, execution.authority_epoch, execution.principal_id, execution.request_id)
    return DeliveryBinding(request.scope, stored, request.message_id,
                           ingress.binding_sha256 if ingress is not None else None,
                           ingress.connector if ingress is not None else "", request.reply_to_message_id)


def encode_binding(binding: DeliveryBinding) -> str:
    return json.dumps(asdict(binding), sort_keys=True, separators=(",", ":"))


def decode_binding(encoded: str) -> DeliveryBinding:
    record = fields(encoded)
    raw_scope, raw_execution = record.get("scope"), record.get("execution")
    if not isinstance(raw_scope, dict) or not isinstance(raw_execution, dict):
        raise ProbeInputError("delivery_binding")
    source_profile = raw_scope.get("source_profile")
    if source_profile is not None and not isinstance(source_profile, str):
        raise ProbeInputError("source_profile")
    scope = TaskScope(text(raw_scope, "profile_id"), text(raw_scope, "platform"),
                      text(raw_scope, "channel_id"), text(raw_scope, "thread_id"),
                      text(raw_scope, "actor_id"), text(raw_scope, "session_id"), text(raw_scope, "runtime_profile"), source_profile)
    execution = StoredExecution(text(raw_execution, "profile_id"), text(raw_execution, "session_id"),
                                text(raw_execution, "admission_id"), number(raw_execution, "admission_generation"),
                                number(raw_execution, "authority_epoch"), text(raw_execution, "principal_id"), text(raw_execution, "request_id"))
    ingress_hash, reply = record.get("ingress_binding_sha256"), record.get("reply_to_message_id")
    if (ingress_hash is not None and not isinstance(ingress_hash, str)) or (reply is not None and not isinstance(reply, str)):
        raise ProbeInputError("delivery_binding")
    return DeliveryBinding(scope, execution, text(record, "message_id"), ingress_hash, text(record, "connector"), reply)
