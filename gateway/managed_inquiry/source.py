from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .models import digest
from .ocr import LocalOcr, OcrReader, _file_hash
from .source_records import Conversation, ReadOnlyRecords, failure_messages
from .task_models import (
    DeliveryReference,
    Finding,
    InquiryRequest,
    Inspection,
    SourceLink,
)


def normalize(text: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKC", text).casefold() if character.isalnum())


def _matches(conversation: Conversation, hint: str) -> bool:
    anchors = [(item.timestamp, "user", normalize(item.text)) for item in conversation.inputs if len(normalize(item.text)) >= 4]
    anchors.extend((message.timestamp, "assistant", normalize(message.content)) for message in conversation.messages if message.role == "assistant" and not message.tool_calls and len(normalize(message.content)) >= 64)
    offset = 0
    user_inputs: set[str] = set()
    assistant_matched = False
    for _timestamp, role, text in sorted(anchors):
        found = hint.find(text, offset)
        if found < 0:
            continue
        offset = found + len(text)
        if role == "user":
            user_inputs.add(text)
        else:
            assistant_matched = True
    return len(user_inputs) >= 2 and assistant_matched


@dataclass(frozen=True, slots=True)
class SQLiteInquirySource:
    db_path: Path
    ocr: OcrReader | None = None

    def find(self, request: InquiryRequest, reference: DeliveryReference | None = None) -> SourceLink | None:
        ingress = request.ingress
        if ingress is None or request.control.expired or request.execution is None:
            return None
        try:
            linked = request.reply_to_message_id is not None and ingress.reply_to_message_id == request.reply_to_message_id
            images = tuple(ingress.current_images)
            if not linked and (ingress.caption_present or not images or len(images) > 4 or self.ocr is None):
                return None
            hints: list[str] = []
            if not linked:
                for attachment in images:
                    if attachment.origin != "current" or attachment.source_message_id != request.message_id or not attachment.mime.startswith("image/"):
                        return None
                    text = self.ocr.read(attachment, request.control) if self.ocr is not None else None
                    if text is None or request.control.expired:
                        return None
                    hints.append(text)
            hint = normalize("\n".join(hints))
            candidates = ReadOnlyRecords(self.db_path).conversations(request, reference if linked else None)
            matching = [item for item in candidates if linked or _matches(item, hint)]
            if len(matching) != 1 or (not linked and not failure_messages(matching[0].messages)) or request.control.expired:
                return None
            selected = matching[0]
            image_sha = images[0].sha256 if len(images) == 1 else digest(json.dumps(sorted(item.sha256 for item in images), separators=(",", ":")).encode()) if images else None
            return SourceLink(request.scope.key, request.identity_sha256, selected.session_id,
                              tuple(item.evidence for item in selected.inputs),
                              tuple(message.evidence for message in selected.messages), image_sha)
        except (OSError, sqlite3.Error, ValueError, RecursionError):
            return None

    def inspect(self, request: InquiryRequest, link: SourceLink) -> Inspection:
        if link.scope_key != request.scope.key or link.request_sha256 != request.identity_sha256 or request.control.expired:
            return Inspection(False, "")
        try:
            messages = ReadOnlyRecords(self.db_path).inspect(request, link)
            if messages is None or not failure_messages(messages):
                return Inspection(False, link.evidence_sha256)
            profile_root = self.db_path.parent
            paths = (profile_root / "config.yaml", profile_root / "skills/bluemango-ops-inquiry/scripts/ops_status_probe.py", Path(__file__).with_name("probe.py"))
            source_hashes = tuple(_file_hash(path, 1_000_000)[0] for path in paths)
            if request.control.expired:
                return Inspection(False, link.evidence_sha256)
            evidence = digest(json.dumps((link.evidence_sha256, source_hashes), separators=(",", ":")).encode())
            findings = (Finding.HEALTH_PARSE_FAILED, Finding.MIGRATION_HTML)
            if not any("ops_status_probe" in (message.tool_calls or "") for message in messages):
                findings += (Finding.RECOVERY_NOT_OBSERVED,)
            return Inspection(True, evidence, findings)
        except (OSError, sqlite3.Error, ValueError, RecursionError):
            return Inspection(False, link.evidence_sha256)


__all__ = ["LocalOcr", "SQLiteInquirySource"]
