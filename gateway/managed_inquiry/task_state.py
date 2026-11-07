from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import uuid
from collections.abc import Generator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .models import ProbeInputError, digest
from .renderer import render
from .task_codec import (
    decode_binding,
    decode_completion,
    decode_link,
    delivery_binding,
    encode_binding,
    encode_completion,
    encode_link,
)
from .task_models import (
    CompletionDraft,
    CompletionState,
    DeliveryReference,
    DeliverySnapshot,
    InquiryRequest,
    PendingTask,
    PreparedCompletion,
    TaskKind,
    TaskScope,
    TaskSelection,
    WorkClaim,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS roots (
 scope_key TEXT PRIMARY KEY, group_key TEXT NOT NULL, task_id TEXT NOT NULL,
 kind TEXT NOT NULL, pending_reset_epoch INTEGER NOT NULL, active INTEGER NOT NULL,
 source_json TEXT, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prepared (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, admission_id TEXT UNIQUE NOT NULL,
 scope_key TEXT NOT NULL, request_sha256 TEXT NOT NULL, token TEXT NOT NULL,
 request_binding_json TEXT NOT NULL, source_json TEXT, result_json TEXT,
 completion_id TEXT UNIQUE, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_refs (
 platform_message_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
 actor_id TEXT NOT NULL, platform TEXT NOT NULL, admission_id TEXT NOT NULL,
 request_id TEXT NOT NULL, source_session_id TEXT NOT NULL,
 channel_id TEXT NOT NULL, thread_id TEXT NOT NULL, principal_id TEXT NOT NULL,
 connector TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class PendingStore:
    path: Path

    @contextmanager
    def _connection(self, write: bool) -> Generator[sqlite3.Connection]:
        if self.path.is_symlink():
            raise ProbeInputError("state_symlink")
        if write:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ProbeInputError("state_file")
            uri = str(self.path)
        else:
            uri = self.path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, timeout=2.0, uri=not write)) as connection:
            connection.row_factory = sqlite3.Row
            if write:
                connection.executescript(_SCHEMA)
                connection.execute("BEGIN IMMEDIATE")
            else:
                connection.execute("PRAGMA query_only=ON")
            with connection:
                yield connection

    def cached(self, request: InquiryRequest) -> PreparedCompletion | None:
        if not self.path.exists() or request.execution is None:
            return None
        with self._connection(False) as connection:
            row = connection.execute("SELECT result_json,request_sha256 FROM prepared WHERE admission_id=?", (request.execution.admission_id,)).fetchone()
            if row is None or row["result_json"] is None:
                return None
            if row["request_sha256"] != request.identity_sha256:
                raise ProbeInputError("completion_identity")
            return decode_completion(row["result_json"])

    def pending(self, scope: TaskScope) -> PendingTask | None:
        if not self.path.exists():
            return None
        with self._connection(False) as connection:
            row = connection.execute("SELECT * FROM roots WHERE scope_key=? AND active=1", (scope.key,)).fetchone()
            return self._task(row) if row is not None else None

    @staticmethod
    def _task(row: sqlite3.Row) -> PendingTask:
        return PendingTask(row["task_id"], TaskKind(row["kind"]), row["pending_reset_epoch"], decode_link(row["source_json"]))

    def begin(self, request: InquiryRequest, selection: TaskSelection) -> WorkClaim | PreparedCompletion | None:
        execution = request.execution
        if execution is None:
            raise ProbeInputError("execution")
        with self._connection(True) as connection:
            old = connection.execute("SELECT * FROM prepared WHERE admission_id=?", (execution.admission_id,)).fetchone()
            if old is not None:
                if old["request_sha256"] != request.identity_sha256:
                    raise ProbeInputError("completion_identity")
                if old["result_json"] is not None:
                    return decode_completion(old["result_json"])
                row = connection.execute("SELECT * FROM roots WHERE scope_key=?", (request.scope.key,)).fetchone()
                if row is None:
                    raise ProbeInputError("pending_missing")
                return WorkClaim(request, self._task(row), None)
            if request.control.expired:
                return None
            connection.execute("UPDATE roots SET active=0,pending_reset_epoch=pending_reset_epoch+1 WHERE group_key=? AND scope_key<>? AND active=1", (request.scope.group_key, request.scope.key))
            row = connection.execute("SELECT * FROM roots WHERE scope_key=?", (request.scope.key,)).fetchone()
            epoch = row["pending_reset_epoch"] if row is not None else 0
            if selection.reset:
                epoch += 1
            if row is None or not row["active"] or selection.reset:
                task = PendingTask(uuid.uuid4().hex, selection.kind, epoch, selection.source)
                connection.execute("INSERT OR REPLACE INTO roots VALUES(?,?,?,?,?,1,?,?)", (request.scope.key, request.scope.group_key, task.task_id, task.kind.value, epoch, encode_link(task.source), time.time()))
            else:
                task = self._task(row)
                if selection.source is not None:
                    task = replace(task, source=selection.source)
                    connection.execute("UPDATE roots SET source_json=? WHERE scope_key=?", (encode_link(task.source), request.scope.key))
            token = uuid.uuid4().hex
            connection.execute("INSERT INTO prepared(admission_id,scope_key,request_sha256,token,request_binding_json,source_json,created_at) VALUES(?,?,?,?,?,?,?)", (execution.admission_id, request.scope.key, request.identity_sha256, token, encode_binding(delivery_binding(request)), encode_link(task.source), time.time()))
            return WorkClaim(request, task, token)

    def finish(self, claim: WorkClaim, draft: CompletionDraft) -> PreparedCompletion:
        request = claim.request
        execution = request.execution
        if execution is None:
            raise ProbeInputError("execution")
        with self._connection(True) as connection:
            row = connection.execute("SELECT * FROM prepared WHERE admission_id=?", (execution.admission_id,)).fetchone()
            if row is None or row["request_sha256"] != request.identity_sha256:
                raise ProbeInputError("completion_identity")
            if row["result_json"] is not None:
                return decode_completion(row["result_json"])
            if row["token"] != claim.token:
                raise ProbeInputError("completion_owner")
            root = connection.execute("SELECT * FROM roots WHERE scope_key=?", (request.scope.key,)).fetchone()
            if root is None or root["task_id"] != claim.task.task_id or root["pending_reset_epoch"] != claim.task.pending_reset_epoch:
                raise ProbeInputError("stale_pending")
            chosen = render(CompletionState.TIMED_OUT, "") if request.control.expired and draft.state not in {CompletionState.TIMED_OUT, CompletionState.HANDLER_ERROR} else draft
            completion_id = digest((request.identity_sha256 + claim.task.task_id + str(claim.task.pending_reset_epoch)).encode())
            result = PreparedCompletion(completion_id, request.scope.key, claim.task.task_id, claim.task.kind, claim.task.pending_reset_epoch, execution.admission_id, execution.admission_generation, execution.authority_epoch, request.identity_sha256, chosen.state, chosen.text, chosen.retains_pending, chosen.evidence_sha256, "", chosen.detail_code)
            receipt = digest(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")).encode())
            result = replace(result, receipt_sha256=receipt)
            connection.execute("UPDATE prepared SET result_json=?,completion_id=? WHERE admission_id=? AND result_json IS NULL", (encode_completion(result), result.completion_id, execution.admission_id))
            connection.execute("UPDATE roots SET active=?,updated_at=? WHERE scope_key=?", (int(result.retains_pending), time.time(), request.scope.key))
            connection.execute("DELETE FROM delivery_refs WHERE admission_id IN (SELECT admission_id FROM prepared ORDER BY seq DESC LIMIT -1 OFFSET 512)")
            connection.execute("DELETE FROM prepared WHERE result_json IS NOT NULL AND seq IN (SELECT seq FROM prepared ORDER BY seq DESC LIMIT -1 OFFSET 512)")
            connection.execute("DELETE FROM roots WHERE active=0 AND scope_key IN (SELECT scope_key FROM roots ORDER BY updated_at DESC LIMIT -1 OFFSET 128)")
            return result

    def abort(self, request: InquiryRequest, state: CompletionState = CompletionState.TIMED_OUT) -> PreparedCompletion | None:
        if state not in {CompletionState.TIMED_OUT, CompletionState.HANDLER_ERROR}:
            raise ProbeInputError("abort_state")
        request.control.cancel()
        if request.execution is None or not self.path.exists():
            return None
        cached = self.cached(request)
        if cached is not None:
            return cached
        with self._connection(False) as connection:
            row = connection.execute("SELECT token FROM prepared WHERE admission_id=? AND request_sha256=?", (request.execution.admission_id, request.identity_sha256)).fetchone()
        task = self.pending(request.scope)
        if row is None or task is None:
            return None
        return self.finish(WorkClaim(request, task, row["token"]), render(state, ""))

    def verify(self, completion: PreparedCompletion, request: InquiryRequest) -> bool:
        if not self.path.exists() or request.execution is None or completion.scope_key != request.scope.key or completion.request_sha256 != request.identity_sha256:
            return False
        with self._connection(False) as connection:
            row = connection.execute("SELECT result_json FROM prepared WHERE admission_id=? AND request_sha256=?", (completion.admission_id, request.identity_sha256)).fetchone()
            root = connection.execute("SELECT task_id,pending_reset_epoch FROM roots WHERE scope_key=?", (completion.scope_key,)).fetchone()
            return bool(row is not None and root is not None and row["result_json"] == encode_completion(completion) and root["task_id"] == completion.task_id and root["pending_reset_epoch"] == completion.pending_reset_epoch)

    def lookup_delivery_completion(self, completion_id: str, scope: TaskScope, text: str) -> DeliverySnapshot | None:
        if not self.path.exists() or len(completion_id) != 64:
            return None
        with self._connection(False) as connection:
            row = connection.execute("SELECT * FROM prepared WHERE completion_id=? AND scope_key=? AND result_json IS NOT NULL", (completion_id, scope.key)).fetchone()
            if row is None:
                return None
            completion = decode_completion(row["result_json"])
            binding = decode_binding(row["request_binding_json"])
            root = connection.execute("SELECT task_id,pending_reset_epoch FROM roots WHERE scope_key=?", (scope.key,)).fetchone()
            if binding.scope != scope or binding.identity_sha256 != completion.request_sha256 or completion.completion_id != completion_id or completion.text != text or root is None or root["task_id"] != completion.task_id or root["pending_reset_epoch"] != completion.pending_reset_epoch:
                return None
            return DeliverySnapshot(completion, binding, decode_link(row["source_json"]))

    def record_delivery_replay(self, snapshot: DeliverySnapshot, platform_message_id: str) -> None:
        completion, binding = snapshot.completion, snapshot.binding
        current = self.lookup_delivery_completion(completion.completion_id, binding.scope, completion.text)
        if not platform_message_id or len(platform_message_id) > 128 or current != snapshot:
            raise ProbeInputError("delivery_replay")
        with self._connection(True) as connection:
            root = connection.execute("SELECT task_id,pending_reset_epoch FROM roots WHERE scope_key=?", (binding.scope.key,)).fetchone()
            if root is None or root["task_id"] != completion.task_id or root["pending_reset_epoch"] != completion.pending_reset_epoch:
                raise ProbeInputError("stale_delivery_reference")
            old = connection.execute("SELECT admission_id FROM delivery_refs WHERE platform_message_id=?", (platform_message_id,)).fetchone()
            if old is not None and old["admission_id"] != completion.admission_id:
                raise ProbeInputError("delivery_reference_conflict")
            connection.execute("INSERT OR IGNORE INTO delivery_refs VALUES(?,?,?,?,?,?,?,?,?,?,?)", (platform_message_id, binding.scope.profile_id, binding.scope.actor_id, binding.scope.platform, completion.admission_id, binding.execution.request_id, binding.scope.session_id, binding.scope.channel_id, binding.scope.thread_id, binding.execution.principal_id, binding.connector))

    def record_delivery_reference(self, completion: PreparedCompletion, request: InquiryRequest, platform_message_id: str) -> None:
        if not platform_message_id or len(platform_message_id) > 128 or not self.verify(completion, request) or request.execution is None:
            raise ProbeInputError("delivery_reference")
        with self._connection(True) as connection:
            root = connection.execute("SELECT task_id,pending_reset_epoch FROM roots WHERE scope_key=?", (completion.scope_key,)).fetchone()
            stored = connection.execute("SELECT result_json FROM prepared WHERE admission_id=?", (completion.admission_id,)).fetchone()
            if root is None or root["task_id"] != completion.task_id or root["pending_reset_epoch"] != completion.pending_reset_epoch or stored is None or stored["result_json"] != encode_completion(completion):
                raise ProbeInputError("stale_delivery_reference")
            old = connection.execute("SELECT admission_id FROM delivery_refs WHERE platform_message_id=?", (platform_message_id,)).fetchone()
            if old is not None and old["admission_id"] != completion.admission_id:
                raise ProbeInputError("delivery_reference_conflict")
            connection.execute("INSERT OR IGNORE INTO delivery_refs VALUES(?,?,?,?,?,?,?,?,?,?,?)", (platform_message_id, request.scope.profile_id, request.scope.actor_id, request.scope.platform, completion.admission_id, request.execution.request_id, request.scope.session_id, request.scope.channel_id, request.scope.thread_id, request.execution.principal_id, request.ingress.connector if request.ingress is not None else ""))

    def lookup_reply(self, request: InquiryRequest) -> DeliveryReference | None:
        ingress, execution = request.ingress, request.execution
        if not request.reply_to_message_id or ingress is None or execution is None or ingress.reply_to_message_id != request.reply_to_message_id or ingress.profile_id != request.scope.profile_id or ingress.actor_id != request.scope.actor_id or not self.path.exists():
            return None
        with self._connection(False) as connection:
            row = connection.execute("SELECT d.admission_id,d.request_id,d.source_session_id,p.result_json FROM delivery_refs d JOIN prepared p ON p.admission_id=d.admission_id WHERE d.platform_message_id=? AND d.profile_id=? AND d.actor_id=? AND d.platform=? AND d.channel_id=? AND d.thread_id=? AND d.principal_id=? AND d.connector=?", (request.reply_to_message_id, request.scope.profile_id, request.scope.actor_id, request.scope.platform, request.scope.channel_id, request.scope.thread_id, execution.principal_id, ingress.connector)).fetchone()
            if row is not None:
                completion = decode_completion(row["result_json"])
                root = connection.execute("SELECT task_id,pending_reset_epoch FROM roots WHERE scope_key=?", (completion.scope_key,)).fetchone()
                if root is None or root["task_id"] != completion.task_id or root["pending_reset_epoch"] != completion.pending_reset_epoch:
                    return None
            return DeliveryReference(row["admission_id"], row["request_id"], row["source_session_id"]) if row is not None else None
