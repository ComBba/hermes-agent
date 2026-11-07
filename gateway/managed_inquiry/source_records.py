from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .models import JsonValue, ProbeInputError, digest
from .task_models import (
    DeliveryReference,
    InquiryRequest,
    SourceAdmission,
    SourceLink,
    SourceMessage,
)

MAX_ADMISSIONS = 256
MAX_MESSAGES = 128
MAX_RECORD_BYTES = 128_000
SOURCE_WINDOW_SECONDS = 900
_SOURCE_ACTOR = "json_extract(CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END,'$.native_text_v1.source.user_id')"


@dataclass(frozen=True, slots=True)
class NativeInput:
    session_id: str
    evidence: SourceAdmission
    text: str
    timestamp: float
    message_id: str | None


@dataclass(frozen=True, slots=True)
class Message:
    row_id: int
    role: str
    content: str
    timestamp: float
    tool_call_id: str | None
    tool_calls: str | None
    tool_name: str | None

    @property
    def evidence(self) -> SourceMessage:
        call = (self.tool_call_id, self.tool_calls, self.tool_name)
        return SourceMessage(self.row_id, self.role, digest(self.content.encode("utf-8", "surrogatepass")),
                             digest(json.dumps(call, separators=(",", ":")).encode()) if any(call) else None)


@dataclass(frozen=True, slots=True)
class Conversation:
    session_id: str
    inputs: tuple[NativeInput, ...]
    messages: tuple[Message, ...]


def event_time(request: InquiryRequest) -> float:
    if request.ingress is None:
        raise ProbeInputError("source_provenance")
    timestamp = dt.datetime.fromisoformat(request.ingress.event_timestamp)
    if timestamp.tzinfo is None:
        raise ProbeInputError("source_timestamp")
    return timestamp.timestamp()


def _native(row: sqlite3.Row, request: InquiryRequest) -> NativeInput | None:
    if row["payload_json"] is None or request.execution is None:
        raise ProbeInputError("source_size")
    payload: JsonValue = json.loads(row["payload_json"])
    if not isinstance(payload, dict):
        return None
    input_text = payload.get("text")
    if not isinstance(input_text, str):
        return None
    envelope = payload.get("native_text_v1")
    if not isinstance(envelope, dict):
        return None
    source, provenance, event = envelope.get("source"), envelope.get("provenance"), envelope.get("event")
    if not isinstance(source, dict) or not isinstance(provenance, dict) or not isinstance(event, dict):
        return None
    if source.get("profile") != request.scope.source_profile or source.get("user_id") != request.scope.actor_id or source.get("platform") != request.scope.platform:
        return None
    profile, channel, thread = source.get("profile"), source.get("chat_id"), source.get("thread_id")
    if (profile is not None and not isinstance(profile, str)) or not isinstance(channel, str) or (thread is not None and not isinstance(thread, str)):
        return None
    principal = "messaging:" + json.dumps([profile, source["platform"], channel, thread, source["user_id"]], separators=(",", ":"))
    if row["principal_id"] != principal:
        return None
    skills = event.get("auto_skill")
    if not isinstance(skills, list) or "bluemango-ops-inquiry" not in skills:
        return None
    ingress = request.ingress
    if ingress is None:
        return None
    if any(provenance.get(key) != value for key, value in (
        ("runtime_home", ingress.runtime_home), ("transport_home", ingress.transport_home),
        ("transport_profile", ingress.transport_profile), ("connector", ingress.connector),
        ("platform", ingress.platform),
    )):
        return None
    stamped = envelope.get("timestamp")
    if not isinstance(stamped, str):
        raise ProbeInputError("source_timestamp")
    timestamp = dt.datetime.fromisoformat(stamped)
    if timestamp.tzinfo is None or type(row["generation"]) is not int or type(row["owner_epoch"]) is not int:
        raise ProbeInputError("source_execution")
    encoded = json.dumps({"target": row["target_session_id"], "payload": {"input": payload, "intent": row["intent"]}}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8", "surrogatepass")
    if digest(encoded) != row["payload_digest"]:
        raise ProbeInputError("source_fingerprint")
    message_id = event.get("message_id")
    if message_id is not None and not isinstance(message_id, str):
        raise ProbeInputError("source_message")
    evidence = SourceAdmission(row["admission_id"], row["generation"], row["owner_epoch"], row["payload_digest"], principal)
    return NativeInput(row["target_session_id"], evidence, input_text, timestamp.timestamp(), message_id)


def _message(row: sqlite3.Row) -> Message:
    if (row["content"] is None and row["content_size"] > MAX_RECORD_BYTES) or (row["tool_calls"] is None and row["calls_size"] > MAX_RECORD_BYTES):
        raise ProbeInputError("source_message_size")
    return Message(row["id"], row["role"], row["content"] or "", row["timestamp"], row["tool_call_id"], row["tool_calls"], row["tool_name"])


def failure_messages(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    calls: set[str] = set()
    for message in messages:
        if message.tool_calls is None:
            continue
        decoded: JsonValue = json.loads(message.tool_calls)
        if not isinstance(decoded, list):
            continue
        for call in decoded:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or function.get("name") != "terminal":
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            if any(client in arguments for client in ("curl", "urlopen", "urllib.request", "requests.get")):
                call_id = call["id"]
                if isinstance(call_id, str):
                    calls.add(call_id)
    failures = tuple(message for message in messages if message.role == "tool" and message.tool_call_id in calls)
    parse_failed = any("JSONDecodeError" in message.content for message in failures)
    moved = any("html" in message.content.casefold() and ("주소 변경" in message.content or "moved" in message.content.casefold()) for message in failures)
    return failures if parse_failed and moved else ()


@dataclass(frozen=True, slots=True)
class ReadOnlyRecords:
    path: Path

    def _connection(self, request: InquiryRequest) -> sqlite3.Connection:
        if request.control.expired or request.ingress is None or self.path.parent.resolve() != Path(request.ingress.runtime_home).resolve():
            raise ProbeInputError("source_scope")
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(request.control.expired), 1000)
        return connection

    @staticmethod
    def _current(connection: sqlite3.Connection, request: InquiryRequest) -> sqlite3.Row:
        execution = request.execution
        if execution is None or execution.session_id != request.scope.session_id or execution.profile_id != request.scope.profile_id:
            raise ProbeInputError("source_admission")
        current = connection.execute("SELECT seq,generation,owner_epoch,principal_id,target_session_id FROM session_admissions WHERE admission_id=?", (execution.admission_id,)).fetchone()
        if current is None or current["principal_id"] != execution.principal_id or current["target_session_id"] != request.scope.session_id or current["generation"] != execution.admission_generation or current["owner_epoch"] != execution.authority_epoch:
            raise ProbeInputError("source_admission")
        return current

    def conversations(self, request: InquiryRequest, reference: DeliveryReference | None = None) -> tuple[Conversation, ...]:
        if request.execution is None:
            return ()
        upper = event_time(request)
        lower = upper - SOURCE_WINDOW_SECONDS
        with closing(self._connection(request)) as connection:
            current = self._current(connection, request)
            selected_seq: int | None = None
            if reference is not None or request.reply_to_message_id is not None:
                if request.ingress is None or request.ingress.reply_to_message_id != request.reply_to_message_id:
                    return ()
                linked = connection.execute(f"SELECT seq FROM session_admissions WHERE {_SOURCE_ACTOR}=? AND seq<? AND status='terminal' AND (admission_id=? OR request_id=?) LIMIT 2", (request.scope.actor_id, current["seq"], reference.admission_id if reference is not None else "", reference.request_id if reference is not None else request.reply_to_message_id)).fetchall()
                if len(linked) != 1:
                    return ()
                selected_seq = linked[0]["seq"]
            min_seq = max(0, selected_seq - MAX_ADMISSIONS // 2) if selected_seq is not None else 0
            max_seq = min(current["seq"], selected_seq + MAX_ADMISSIONS // 2) if selected_seq is not None else current["seq"]
            rows = connection.execute(f"SELECT seq,admission_id,principal_id,target_session_id,CASE WHEN length(CAST(payload_json AS BLOB))<=? THEN payload_json ELSE NULL END AS payload_json,payload_digest,intent,generation,owner_epoch FROM session_admissions WHERE {_SOURCE_ACTOR}=? AND seq<? AND seq>=? AND status='terminal' ORDER BY seq DESC LIMIT ?", (MAX_RECORD_BYTES, request.scope.actor_id, max_seq, min_seq, MAX_ADMISSIONS + 1)).fetchall()
            if len(rows) > MAX_ADMISSIONS:
                raise ProbeInputError("source_admission_limit")
            parsed_inputs: list[NativeInput] = []
            selected_native: NativeInput | None = None
            input_sequences: dict[str, int] = {}
            for row in rows:
                if row["payload_json"] is None:
                    raise ProbeInputError("source_size")
                parsed = _native(row, request)
                if parsed is not None:
                    parsed_inputs.append(parsed)
                    input_sequences[parsed.evidence.admission_id] = row["seq"]
                    if row["seq"] == selected_seq:
                        selected_native = parsed
                        lower = parsed.timestamp - SOURCE_WINDOW_SECONDS
                        upper = min(upper, parsed.timestamp + SOURCE_WINDOW_SECONDS)
            inputs = [item for item in parsed_inputs if lower <= item.timestamp <= upper]
            if selected_seq is not None:
                if selected_native is None:
                    return ()
                inputs = [item for item in inputs if item.session_id == selected_native.session_id]
            conversations: list[Conversation] = []
            for session_id in sorted({item.session_id for item in inputs}):
                natives = tuple(sorted((item for item in inputs if item.session_id == session_id), key=lambda item: item.timestamp))
                start_seq = min(input_sequences[item.evidence.admission_id] for item in natives)
                ownership = connection.execute("SELECT principal_id,status FROM session_admissions WHERE target_session_id=? AND seq>=? AND seq<? LIMIT ?", (session_id, start_seq, current["seq"], MAX_ADMISSIONS + 1)).fetchall()
                if len(ownership) != len(natives) or any(row["principal_id"] != natives[0].evidence.principal_id or row["status"] != "terminal" for row in ownership):
                    continue
                messages = connection.execute("SELECT id,role,CASE WHEN length(CAST(content AS BLOB))<=? THEN content ELSE NULL END AS content,length(CAST(COALESCE(content,'') AS BLOB)) AS content_size,timestamp,tool_call_id,CASE WHEN length(CAST(tool_calls AS BLOB))<=? THEN tool_calls ELSE NULL END AS tool_calls,length(CAST(COALESCE(tool_calls,'') AS BLOB)) AS calls_size,tool_name FROM messages WHERE session_id=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp,id LIMIT ?", (MAX_RECORD_BYTES, MAX_RECORD_BYTES, session_id, lower, upper, MAX_MESSAGES + 1)).fetchall()
                if len(messages) > MAX_MESSAGES:
                    raise ProbeInputError("source_message_limit")
                conversations.append(Conversation(session_id, natives, tuple(_message(row) for row in messages)))
            return tuple(conversations)

    def inspect(self, request: InquiryRequest, link: SourceLink) -> tuple[Message, ...] | None:
        if request.execution is None or len(link.source_admissions) > MAX_ADMISSIONS or len(link.messages) > MAX_MESSAGES:
            return None
        with closing(self._connection(request)) as connection:
            current_row = self._current(connection, request)
            if not link.source_admissions or len({item.principal_id for item in link.source_admissions}) != 1:
                return None
            sequences: list[int] = []
            for evidence in link.source_admissions:
                row = connection.execute("SELECT seq,admission_id,principal_id,target_session_id,CASE WHEN length(CAST(payload_json AS BLOB))<=? THEN payload_json ELSE NULL END AS payload_json,payload_digest,intent,generation,owner_epoch FROM session_admissions WHERE admission_id=? AND principal_id=? AND target_session_id=? AND status='terminal' AND seq<?", (MAX_RECORD_BYTES, evidence.admission_id, evidence.principal_id, link.source_session_id, current_row["seq"])).fetchone()
                if row is None:
                    return None
                current = _native(row, request)
                if current is None or current.evidence != evidence:
                    return None
                sequences.append(row["seq"])
            ownership = connection.execute("SELECT admission_id,principal_id,status FROM session_admissions WHERE target_session_id=? AND seq>=? AND seq<? LIMIT ?", (link.source_session_id, min(sequences), current_row["seq"], MAX_ADMISSIONS + 1)).fetchall()
            if {row["admission_id"] for row in ownership} != {item.admission_id for item in link.source_admissions} or any(row["principal_id"] != link.source_admissions[0].principal_id or row["status"] != "terminal" for row in ownership):
                return None
            observed: list[Message] = []
            for evidence in link.messages:
                row = connection.execute("SELECT id,role,CASE WHEN length(CAST(content AS BLOB))<=? THEN content ELSE NULL END AS content,length(CAST(COALESCE(content,'') AS BLOB)) AS content_size,timestamp,tool_call_id,CASE WHEN length(CAST(tool_calls AS BLOB))<=? THEN tool_calls ELSE NULL END AS tool_calls,length(CAST(COALESCE(tool_calls,'') AS BLOB)) AS calls_size,tool_name FROM messages WHERE id=? AND session_id=?", (MAX_RECORD_BYTES, MAX_RECORD_BYTES, evidence.row_id, link.source_session_id)).fetchone()
                message = _message(row) if row is not None else None
                if message is None or message.evidence != evidence:
                    return None
                observed.append(message)
            return tuple(observed)
