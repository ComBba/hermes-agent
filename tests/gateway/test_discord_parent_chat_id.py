"""Every Discord turn in a thread must know its parent channel.

Anything scoped to a parent channel -- a channel prompt, a channel skill set,
a per-channel reasoning policy -- resolves from `parent_chat_id` on the source.
Ordinary messages carried it; slash commands and thread starters did not, even
though both paths already computed the parent id and used it for their own
channel-prompt and skill lookups.

The failure that causes is worse than a setting that never works: the setting
works for messages and silently does not for `/skill`, `/queue`, `/learn` and
the first turn of a new thread. Which turns are affected depends on how each
was started, so it reads as intermittent rather than as a gap.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _ensure_discord_mock() -> None:
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = types.ModuleType("discord")
    discord_mod.Intents = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import discord  # noqa: E402

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


PARENT_CHANNEL_ID = 1528074929583427687
THREAD_CHANNEL_ID = 1533453834909647028


class _Thread(discord.Thread):
    """A thread whose parent is a real channel, as discord.py reports it."""

    def __init__(self) -> None:
        self.id = THREAD_CHANNEL_ID
        self.parent_id = PARENT_CHANNEL_ID
        self.name = "incident-42"
        self.guild = SimpleNamespace(name="ops")


def _interaction(channel: object) -> SimpleNamespace:
    return SimpleNamespace(
        channel=channel,
        channel_id=THREAD_CHANNEL_ID,
        user=SimpleNamespace(id=99, display_name="operator"),
        guild=SimpleNamespace(name="ops"),
    )


@pytest.fixture
def adapter() -> DiscordAdapter:
    instance = DiscordAdapter.__new__(DiscordAdapter)
    # `name` is a read-only property on the adapter; the paths under test do
    # not consult it, so it is left alone rather than forced.
    instance._get_effective_topic = lambda channel, is_thread=False: None
    instance._resolve_channel_prompt = lambda chat, parent=None: None
    instance._resolve_channel_skills = lambda chat, parent=None: None
    instance._thread_parent_channel = lambda channel: SimpleNamespace(
        id=PARENT_CHANNEL_ID
    )
    captured: dict[str, object] = {}

    def build_source(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    instance.build_source = build_source
    instance.captured = captured
    return instance


def test_slash_command_in_a_thread_carries_its_parent(adapter) -> None:
    event = adapter._build_slash_event(_interaction(_Thread()), "/skill list")
    assert adapter.captured["parent_chat_id"] == str(PARENT_CHANNEL_ID)
    assert event.source.parent_chat_id == str(PARENT_CHANNEL_ID)


def test_slash_command_outside_a_thread_reports_no_parent(adapter) -> None:
    # A plain channel has no parent. `None` rather than "" so a consumer can
    # tell "no parent" from "a parent whose id is empty", which would be a bug
    # upstream rather than a fact about the channel.
    channel = SimpleNamespace(id=PARENT_CHANNEL_ID, name="ops", guild=None)
    adapter._build_slash_event(_interaction(channel), "/skill list")
    assert adapter.captured["parent_chat_id"] is None


@pytest.mark.asyncio
async def test_thread_starter_carries_its_parent(adapter) -> None:
    adapter.handle_message = lambda event: None
    interaction = _interaction(_Thread())
    try:
        await adapter._dispatch_thread_session(
            interaction,
            str(THREAD_CHANNEL_ID),
            "incident-42",
            "start here",
        )
    except Exception:
        # The dispatch continues into delivery, which this fixture does not
        # provide. The source is built before any of that, and it is the
        # source this test is about.
        pass
    assert adapter.captured["parent_chat_id"] == str(PARENT_CHANNEL_ID)


def test_every_source_this_adapter_builds_declares_a_parent() -> None:
    """No call site may omit it, including ones added later.

    The two paths fixed here each already computed the parent id and used it
    for their own lookups; only the source was left without it. A new call
    site would be one `build_source(...)` away from the same gap.
    """
    import inspect
    import re

    source = inspect.getsource(
        sys.modules["plugins.platforms.discord.adapter"]
    )
    calls = re.findall(r"self\.build_source\(\s*(.*?)\n\s*\)", source, re.S)
    assert calls, "expected the adapter to build sources"
    missing = [
        call.strip().splitlines()[0]
        for call in calls
        if "parent_chat_id" not in call
    ]
    assert not missing, f"build_source call sites without parent_chat_id: {missing}"
