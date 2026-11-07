from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal


@dataclass(frozen=True, slots=True)
class AttachmentInput:
    attachment: object
    origin: Literal['current', 'reply', 'snapshot']
    source_message_id: str


@dataclass(frozen=True, slots=True)
class VerifiedAttachment:
    attachment_id: str
    origin: str
    source_message_id: str
    mime: str
    sha256: str
    size: int
    local_path: str


@dataclass(frozen=True, slots=True)
class PlatformObservation:
    caption_present: bool
    message_id: str
    actor_id: str
    attachments: tuple[VerifiedAttachment, ...]
    reply_to_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class AdmissionOwner:
    profile_id: str
    session_id: str
    authority_epoch: int


@dataclass(frozen=True, slots=True)
class ExecutionAdmission:
    profile_id: str
    session_id: str
    admission_id: str
    admission_generation: int
    authority_epoch: int
    principal_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class VerifiedIngress:
    schema_version: int
    caption_present: bool
    attachments: tuple[VerifiedAttachment, ...]
    platform: str
    actor_id: str
    profile_id: str
    session_id: str
    message_id: str
    event_timestamp: str
    runtime_home: str
    transport_home: str
    transport_profile: str | None
    connector: str
    admission_id: str
    admission_generation: int
    principal_id: str
    authority_epoch: int
    binding_sha256: str
    request_id: str
    reply_to_message_id: str | None = None

    @property
    def current_images(self) -> tuple[VerifiedAttachment, ...]:
        return tuple(item for item in self.attachments if item.origin == 'current'
                     and item.source_message_id == self.message_id and item.mime.startswith('image/'))


@dataclass(frozen=True, slots=True)
class IngressCheck:
    state: Literal['valid', 'absent', 'invalid']
    provenance: VerifiedIngress | None = None
    reason: str = ''


@dataclass(frozen=True, slots=True)
class _AdmissionContext:
    provenance: VerifiedIngress
    source_sha256: str


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def observe_attachment(item: AttachmentInput, local_path: str, mime: str) -> VerifiedAttachment | None:
    attachment_id = getattr(item.attachment, 'id', None)
    if not isinstance(attachment_id, (str, int)) or isinstance(attachment_id, bool):
        return None
    path = Path(local_path)
    if not path.is_absolute():
        return None
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('platform_media_not_regular')
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('platform_media_changed')
    return VerifiedAttachment(str(attachment_id), item.origin,
                              item.source_message_id, mime, digest, after.st_size, str(path))


def _binding(event, transport) -> str:
    source = event.source
    return _digest({'source': [source.platform.value, source.user_id, source.profile, source.chat_id,
                               source.thread_id, source.scope_id], 'transport': transport,
                    'message_id': event.message_id, 'timestamp': event.timestamp.isoformat(),
                    'reply_to_message_id': event.reply_to_message_id})


def _source_digest(event) -> str:
    source = event.source
    return _digest([source.platform.value, source.user_id, source.profile, source.chat_id,
                    source.thread_id, source.scope_id])


def snapshot_observation(event, transport, media):
    observation = getattr(event, '_managed_platform_observation', None)
    if observation is None:
        return None
    if (not isinstance(observation, PlatformObservation) or transport is None
            or event.source.platform.value != 'discord' or type(observation.caption_present) is not bool
            or observation.message_id != event.message_id or observation.actor_id != event.source.user_id):
        raise ValueError('platform_provenance_untrusted')
    attachments = []
    for item in observation.attachments:
        matches = [reference for reference in media if reference['sha256'] == item.sha256
                   and reference['size'] == item.size
                   and Path(reference['path']).name == Path(item.local_path).name]
        paths = {reference['path'] for reference in matches}
        if len(paths) != 1 or item.local_path not in event.media_urls:
            raise ValueError('platform_media_capture_mismatch')
        attachments.append({**asdict(item), 'local_path': paths.pop()})
    if observation.reply_to_message_id != event.reply_to_message_id:
        raise ValueError('platform_reply_mismatch')
    return {'schema_version': 1, 'caption_present': observation.caption_present,
            'reply_to_message_id': observation.reply_to_message_id,
            'binding_sha256': _binding(event, transport), 'attachments': attachments}


def restore_observation(event, envelope):
    descriptor = envelope.get('platform_ingress_v1')
    if descriptor is None:
        return None
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {'schema_version', 'caption_present', 'binding_sha256', 'attachments', 'reply_to_message_id'}
            or type(descriptor['schema_version']) is not int or descriptor['schema_version'] != 1
            or type(descriptor['caption_present']) is not bool or not isinstance(envelope.get('provenance'), dict)
            or descriptor['binding_sha256'] != _binding(event, envelope.get('provenance'))
            or descriptor['reply_to_message_id'] != event.reply_to_message_id
            or not (descriptor['reply_to_message_id'] is None or isinstance(descriptor['reply_to_message_id'],str))
            or event.source.platform.value != 'discord' or not isinstance(descriptor['attachments'], list)
            or len(descriptor['attachments']) > 100):
        raise ValueError('platform_provenance_invalid')
    attachments = []
    seen = set()
    for row in descriptor['attachments']:
        if (not isinstance(row, dict) or set(row) != set(VerifiedAttachment.__dataclass_fields__)
                or any(not isinstance(row[key], str) or (not row[key] and key != 'source_message_id')
                       for key in row if key != 'size')
                or type(row['size']) is not int or row['size'] < 0
                or row['origin'] not in {'current', 'reply', 'snapshot'}
                or (row['origin'] == 'current' and row['source_message_id'] != event.message_id)
                or (row['attachment_id'], row['origin'], row['source_message_id']) in seen
                or {'path': row['local_path'], 'sha256': row['sha256'], 'size': row['size']}
                not in envelope.get('media', [])):
            raise ValueError('platform_attachment_invalid')
        seen.add((row['attachment_id'], row['origin'], row['source_message_id']))
        attachments.append(VerifiedAttachment(**row))
    return PlatformObservation(descriptor['caption_present'], event.message_id,
                               event.source.user_id, tuple(attachments), descriptor['reply_to_message_id'])


def bind_admission(event, row, owner: AdmissionOwner) -> None:
    envelope = row['payload'].get('native_text_v1')
    if envelope is None:
        return
    if (row['status'] != 'started' or row['owner_epoch'] != owner.authority_epoch
            or type(row['generation']) is not int or row['generation'] < 1
            or (row.get('target_session_id') or row['session_id']) != owner.session_id):
        raise ValueError('execution_admission_invalid')
    event._managed_execution_admission = ExecutionAdmission(
        owner.profile_id, owner.session_id, row['admission_id'], row['generation'],
        owner.authority_epoch, row['principal_id'], row['request_id'])
    if 'platform_ingress_v1' not in envelope:
        return
    observation = restore_observation(event, envelope)
    if observation is None:
        raise ValueError('platform_observation_missing')
    source = event.source
    principal = 'messaging:' + json.dumps([source.profile, source.platform.value, source.chat_id,
                                         source.thread_id, source.user_id], separators=(',', ':'))
    if (row['status'] != 'started' or row['owner_epoch'] != owner.authority_epoch
            or row['principal_id'] != principal or type(row['generation']) is not int
            or row['generation'] < 1 or (row.get('target_session_id') or row['session_id']) != owner.session_id):
        raise ValueError('platform_admission_invalid')
    transport = envelope['provenance']
    receipt = VerifiedIngress(
        1, observation.caption_present, observation.attachments, source.platform.value, source.user_id,
        owner.profile_id, owner.session_id, event.message_id, event.timestamp.isoformat(),
        transport['runtime_home'], transport['transport_home'], transport.get('transport_profile'),
        transport['connector'], row['admission_id'], row['generation'], row['principal_id'],
        owner.authority_epoch, envelope['platform_ingress_v1']['binding_sha256'], row['request_id'],
        observation.reply_to_message_id)
    event._managed_ingress_receipt = _AdmissionContext(receipt, _source_digest(event))


def verified_ingress(event, *, session_id: str, admission_id: str | None = None,
                     admission_generation: int | None = None, authority_epoch: int | None = None) -> IngressCheck:
    context = getattr(event, '_managed_ingress_receipt', None)
    if context is None:
        return IngressCheck('absent')
    if not isinstance(context, _AdmissionContext):
        return IngressCheck('invalid', reason='admission_context_invalid')
    receipt = context.provenance
    execution = execution_admission(event, session_id=session_id)
    if (context.source_sha256 != _source_digest(event) or receipt.session_id != session_id
            or execution is None or execution.admission_id != receipt.admission_id
            or execution.admission_generation != receipt.admission_generation
            or execution.authority_epoch != receipt.authority_epoch
            or execution.request_id != receipt.request_id
            or execution.profile_id != receipt.profile_id
            or receipt.event_timestamp != event.timestamp.isoformat()
            or receipt.actor_id != event.source.user_id or receipt.message_id != event.message_id
            or receipt.reply_to_message_id != event.reply_to_message_id
            or receipt.platform != event.source.platform.value
            or getattr(event, '_claimed_owner', None) != (session_id, receipt.admission_generation)
            or (admission_id is not None and receipt.admission_id != admission_id)
            or (admission_generation is not None and receipt.admission_generation != admission_generation)
            or (authority_epoch is not None and receipt.authority_epoch != authority_epoch)):
        return IngressCheck('invalid', reason='admission_binding_mismatch')
    return IngressCheck('valid', receipt)


def execution_admission(event, *, session_id: str) -> ExecutionAdmission | None:
    execution = getattr(event, '_managed_execution_admission', None)
    if not isinstance(execution, ExecutionAdmission) or event.source is None:
        return None
    source = event.source
    principal = 'messaging:' + json.dumps([source.profile, source.platform.value, source.chat_id,
                                         source.thread_id, source.user_id], separators=(',', ':'))
    if (execution.session_id != session_id or execution.principal_id != principal
            or getattr(event, '_claimed_owner', None) != (session_id, execution.admission_generation)):
        return None
    return execution
