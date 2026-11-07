from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.managed_ingress_provenance import (AdmissionOwner, AttachmentInput, PlatformObservation,
    bind_admission, execution_admission, verified_ingress)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_envelope import prepare_native, restore_native
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import RuntimeStoreError, claim_session_input, list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path/'sessions'))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    try:
        yield runner, adapter, authority
    finally:
        authority.db.close()


async def captured_event(adapter, caption=False):
    observations = []
    inputs = [AttachmentInput(SimpleNamespace(id=identity, filename=f'{identity}.png',
              content_type='image/png', size=40, url='https://invalid.example/image',
              read=AsyncMock(return_value=b'\x89PNG\r\n\x1a\n'+bytes([index])*32)), origin, message)
              for index, (identity, origin, message) in enumerate(
                  [('current-att','current','message'), ('reply-att','reply','previous'),
                   ('snapshot-att','snapshot','forwarded')])]
    urls, types, _, _ = await adapter._collect_attachment_media(inputs, observations=observations)
    event = MessageEvent('[Image attachment]', source=adapter.build_source(chat_id='42', chat_type='dm', user_id='42'),
                         message_id='message', message_type=MessageType.PHOTO,
                         media_urls=urls, media_types=types, timestamp=datetime(2026,11,7,tzinfo=UTC))
    event._managed_platform_observation = PlatformObservation(caption, 'message', '42', tuple(observations))
    return event


@pytest.mark.asyncio
async def test_real_cache_envelope_durable_claim_restore_has_no_raw_platform_dependency(owner):
    runner, adapter, authority = owner
    event = await captured_event(adapter)
    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
    restored = restore_native(row['payload'], runner)
    assert restored.raw_message is None
    bind_admission(restored, row, AdmissionOwner('default', receipt.ref.session_id, authority.epoch))
    restored._claimed_owner = (receipt.ref.session_id, row['generation'])
    result = verified_ingress(restored, session_id=receipt.ref.session_id)
    assert result.state == 'valid'
    proof = result.provenance
    assert proof.caption_present is False
    assert [(a.attachment_id,a.origin) for a in proof.attachments] == [
        ('current-att','current'), ('reply-att','reply'), ('snapshot-att','snapshot')]
    assert [a.attachment_id for a in proof.current_images] == ['current-att']
    assert all(a.sha256 and a.size == 40 for a in proof.attachments)
    assert execution_admission(restored, session_id=receipt.ref.session_id).admission_id == receipt.admission_id
    assert verified_ingress(restored, session_id='other').state == 'invalid'
    assert verified_ingress(restored, session_id=receipt.ref.session_id, admission_generation=row['generation']+1).state == 'invalid'
    assert verified_ingress(restored, session_id=receipt.ref.session_id, authority_epoch=authority.epoch+1).state == 'invalid'
    restored._claimed_owner = (receipt.ref.session_id,row['generation']+1)
    assert verified_ingress(restored,session_id=receipt.ref.session_id).state == 'invalid'
    restored._claimed_owner = (receipt.ref.session_id,row['generation'])
    restored._managed_execution_admission = replace(restored._managed_execution_admission, admission_id='replayed')
    assert verified_ingress(restored, session_id=receipt.ref.session_id).state == 'invalid'


@pytest.mark.asyncio
async def test_callback_required_and_current_cache_bytes_revalidated(owner):
    runner, adapter, _ = owner
    event = await captured_event(adapter)
    with pytest.raises(RuntimeStoreError):
        await prepare_native(runner, event)
    with native_callback(runner, event, get_hermes_home()):
        payload = await prepare_native(runner, event)
    reference = payload['native_text_v1']['media'][0]
    from pathlib import Path
    Path(reference['path']).write_bytes(b'changed')
    with pytest.raises((RuntimeStoreError, ValueError)):
        restore_native(payload, runner)


@pytest.mark.asyncio
async def test_redelivery_keeps_exact_attachment_identity_across_fresh_cache_paths(owner):
    runner, adapter, authority = owner
    first = await captured_event(adapter)
    second = await captured_event(adapter)
    assert first.media_urls != second.media_urls
    with native_callback(runner,first,get_hermes_home()):
        one = await authority.admit_native(first)
    with native_callback(runner,second,get_hermes_home()):
        two = await authority.admit_native(second)
    assert one.admission_id == two.admission_id
    row = list_session_admissions(authority.db,session_id=one.ref.session_id)[0]
    restored = restore_native(row['payload'],runner)
    assert len(restored._managed_platform_observation.attachments) == 3
    changed = await captured_event(adapter)
    changed._managed_platform_observation = replace(changed._managed_platform_observation,caption_present=True)
    with native_callback(runner,changed,get_hermes_home()), pytest.raises(RuntimeStoreError,match='admission_conflict'):
        await authority.admit_native(changed)


@pytest.mark.asyncio
async def test_historical_schema_absent_and_public_metadata_cannot_mint_proof(owner):
    runner, adapter, authority = owner
    event = await captured_event(adapter)
    event._managed_platform_observation = None
    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
    restored = restore_native(row['payload'], runner)
    bind_admission(restored, row, AdmissionOwner('default', receipt.ref.session_id, authority.epoch))
    restored._claimed_owner = (receipt.ref.session_id,row['generation'])
    assert verified_ingress(restored, session_id=receipt.ref.session_id).state == 'absent'
    assert execution_admission(restored, session_id=receipt.ref.session_id) is not None
    event.metadata = {'platform_ingress_v1': {'caption_present':False}}
    with native_callback(runner, event, get_hermes_home()), pytest.raises(RuntimeStoreError):
        await prepare_native(runner, event)


@pytest.mark.asyncio
async def test_native_hold_refuses_before_durable_admission(owner):
    runner,adapter,authority = owner
    runner._external_drain_active = True
    event = await captured_event(adapter)
    with native_callback(runner,event,get_hermes_home()),pytest.raises(RuntimeStoreError,match='runtime_draining'):
        await authority.admit_native(event)
    with authority.db._read_ctx() as connection:
        assert connection.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_closed_descriptor_rejects_projection_and_caption_tampering(owner):
    runner, adapter, _ = owner
    event = await captured_event(adapter)
    with native_callback(runner, event, get_hermes_home()):
        payload = await prepare_native(runner,event)
    for field in ('binding_sha256', 'schema_version', 'caption_present', 'attachments'):
        mutated = deepcopy(payload)
        mutated['native_text_v1']['platform_ingress_v1'].pop(field)
        with pytest.raises(RuntimeStoreError):
            restore_native(mutated, runner)
    mutated = deepcopy(payload)
    mutated['native_text_v1']['event']['message_id'] = 'other-message'
    with pytest.raises(RuntimeStoreError):
        restore_native(mutated,runner)


@pytest.mark.asyncio
@pytest.mark.parametrize('caption', ['', ' ', '[Image attachment]'])
async def test_real_discord_producer_observes_caption_before_snapshot_and_placeholder(owner, monkeypatch, caption):
    _, adapter, _ = owner
    class DM:
        id = 42
        name = 'dm'
    monkeypatch.setattr('plugins.platforms.discord.adapter.discord.DMChannel',DM)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter.handle_message = AsyncMock()
    att = SimpleNamespace(id=123, filename='current.png',content_type='image/png',size=40,
                          url='https://invalid.example/image',
                          read=AsyncMock(return_value=b'\x89PNG\r\n\x1a\n'+b'x'*32))
    message = SimpleNamespace(id=456,content=caption,attachments=[att],mentions=[],reference=None,
        message_snapshots=[SimpleNamespace(content='forwarded caption',attachments=[])],
        channel=DM(),author=SimpleNamespace(id=42,display_name='Fixture',name='Fixture'),
        created_at=datetime(2026,11,7,tzinfo=UTC))
    assert await adapter._handle_message(message)
    event = adapter.handle_message.call_args.args[0]
    proof = event._managed_platform_observation
    assert proof.caption_present is bool(caption)
    assert [(a.attachment_id,a.origin,a.source_message_id) for a in proof.attachments] == [('123','current','456')]
