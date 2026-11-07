from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('name,parent_name', [('123','general'),('#123','general'),('general','123')])
def test_discord_numeric_names_never_impersonate_real_channel_id(name,parent_name):
    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=456,name=name,parent_id=789,parent=SimpleNamespace(name=parent_name))
    keys = adapter._discord_channel_keys_from_channel(channel)
    assert {'456','789'} <= keys
    assert '123' not in keys and '#123' not in keys
    assert 'general' in keys and '#general' in keys


@pytest.mark.parametrize('platform,chat,thread,expected', [
    ('discord','parent',None,True),('discord','thread',None,False),
    ('discord','elsewhere',None,False),('telegram','parent',None,False),
    ('discord','parent','explicit',False)])
def test_cron_origin_thread_warning_only_matches_same_parent(monkeypatch,caplog,platform,chat,thread,expected):
    import logging
    from cron import scheduler_delivery as delivery
    monkeypatch.setattr(delivery,'_resolve_target_transport',lambda *args:(None,'fixture_unavailable'))
    target = {'platform':platform,'chat_id':chat,'thread_id':thread}
    job = {'id':'fixture','origin':{'platform':'discord','chat_id':'parent','thread_id':'thread'}}
    monkeypatch.setattr(delivery,'_resolve_origin',lambda value:value['origin'])
    with caplog.at_level(logging.WARNING):
        delivery._prepare_target_delivery(job,target,adapters={},loop=None,config=None,
            notify_delivery=False,mirror_enabled=False,mirror_text='',delivery_errors=[])
    assert ('delivery target lost it' in caplog.text) is expected


def test_ocr_disabled_and_failed_reading_never_becomes_confirmation(monkeypatch):
    from agent import image_routing as routing
    monkeypatch.setattr(routing,'_ocr_command',list)
    assert routing.ocr_crosscheck_hint(b'image') is None
    monkeypatch.setattr(routing,'_ocr_command',lambda:['fixture-ocr'])
    monkeypatch.setattr(routing,'_run_ocr',lambda *args:(None,'fixture_failure'))
    hint = routing.ocr_crosscheck_hint(b'image')
    assert 'unverifiable' in hint.lower() or 'unavailable' in hint.lower()
    monkeypatch.setattr(routing,'_run_ocr',lambda *args:(['A8','AB'],'transition'))
    hint = routing.ocr_crosscheck_hint(b'image')
    assert 'A8' in hint and 'AB' in hint
    assert routing.OCR_CROSSCHECK_RULE in hint
