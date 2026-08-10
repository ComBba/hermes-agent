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

These exercise the real `build_source` -- the one that constructs a
`SessionSource` and resolves the profile route -- rather than a stand-in, so
what is asserted is the source a turn actually receives.
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

from gateway.config import Platform  # noqa: E402
from gateway.platforms.base import BasePlatformAdapter  # noqa: E402
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
    """A Discord adapter using the real `build_source`.

    Only the collaborators these two paths reach are stubbed -- topic
    resolution and the channel prompt/skill lookups, which read configuration
    this test says nothing about. `build_source` itself is the inherited
    implementation, so `SessionSource` construction and profile-route
    resolution run for real.
    """
    instance = DiscordAdapter.__new__(DiscordAdapter)
    instance.build_source = BasePlatformAdapter.build_source.__get__(
        instance, DiscordAdapter
    )
    instance.platform = Platform.DISCORD
    instance.gateway_runner = None
    instance._get_effective_topic = lambda channel, is_thread=False: None
    instance._resolve_channel_prompt = lambda chat, parent=None: None
    instance._resolve_channel_skills = lambda chat, parent=None: None
    instance._thread_parent_channel = lambda channel: SimpleNamespace(
        id=PARENT_CHANNEL_ID
    )
    return instance


def test_slash_command_in_a_thread_carries_its_parent(adapter) -> None:
    event = adapter._build_slash_event(_interaction(_Thread()), "/skill list")
    assert event.source.parent_chat_id == str(PARENT_CHANNEL_ID)
    assert event.source.thread_id == str(THREAD_CHANNEL_ID)
    assert event.source.platform == Platform.DISCORD


def test_slash_command_outside_a_thread_reports_no_parent(adapter) -> None:
    # A plain channel has no parent, and the source says so as `None`.
    # `build_source` owns that normalisation -- it already maps a falsy value
    # to `None` -- so these paths pass the id they have and do not repeat it.
    channel = SimpleNamespace(id=PARENT_CHANNEL_ID, name="ops", guild=None)
    event = adapter._build_slash_event(_interaction(channel), "/skill list")
    assert event.source.parent_chat_id is None


@pytest.mark.asyncio
async def test_thread_starter_carries_its_parent(adapter) -> None:
    delivered: list[object] = []

    async def handle_message(event: object) -> None:
        delivered.append(event)

    adapter.handle_message = handle_message
    await adapter._dispatch_thread_session(
        _interaction(_Thread()),
        str(THREAD_CHANNEL_ID),
        "incident-42",
        "start here",
    )

    assert len(delivered) == 1, "the starter must reach handle_message"
    source = delivered[0].source
    assert source.parent_chat_id == str(PARENT_CHANNEL_ID)
    assert source.thread_id == str(THREAD_CHANNEL_ID)


@pytest.mark.asyncio
async def test_a_thread_turn_resolves_the_same_parent_however_it_started(
    adapter,
) -> None:
    """The invariant, rather than three separate expected values.

    A slash command and a thread starter in the same thread describe the same
    conversation. Whatever a consumer resolves from the parent must not depend
    on which of them began the turn -- that dependence is the defect, and it
    is what made the gap read as intermittent.
    """
    delivered: list[object] = []

    async def handle_message(event: object) -> None:
        delivered.append(event)

    adapter.handle_message = handle_message
    await adapter._dispatch_thread_session(
        _interaction(_Thread()),
        str(THREAD_CHANNEL_ID),
        "incident-42",
        "start here",
    )
    slash = adapter._build_slash_event(_interaction(_Thread()), "/skill list")

    assert (
        delivered[0].source.parent_chat_id == slash.source.parent_chat_id
    ), "the same thread must resolve the same parent from either entry point"
