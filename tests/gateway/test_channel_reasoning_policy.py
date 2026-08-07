from types import SimpleNamespace

import pytest

from gateway.channel_reasoning import resolve_channel_reasoning_config


DEV = "1528074929583427687"
RESERVATION = "1533453834909647028"


@pytest.fixture
def config():
    return {
        "discord": {
            "channel_reasoning_policies": {
                DEV: {
                    "default_effort": "high",
                    "xhigh_terms": ["architecture", "근본 원인", "복합 장애"],
                    "ultra": {
                        "enabled": True,
                        "terms": ["병렬", "parallel"],
                        "requires_thread": True,
                        "blocked_terms": ["배포", "삭제", "재시작", "운영 변경"],
                    },
                },
                RESERVATION: {
                    "default_effort": "medium",
                    "high_terms": ["분석", "전략", "계획서", "보고서", "docx", "pdf", "xlsx"],
                    "xhigh_terms": ["근본 원인", "복합 진단", "교차 검증", "최종 검토"],
                    "ultra": {"enabled": False},
                },
            }
        }
    }


def resolve(config, channel, message, *, thread_id="thread-1"):
    return resolve_channel_reasoning_config(
        config,
        platform="discord",
        chat_id=channel,
        thread_id=thread_id,
        parent_id=channel,
        message=message,
    )


def test_development_defaults_to_high(config):
    assert resolve(config, DEV, "현재 상태를 확인해줘") == {"enabled": True, "effort": "high"}


def test_development_complex_raises_to_xhigh(config):
    assert resolve(config, DEV, "복합 장애의 근본 원인을 분석해줘") == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_development_explicit_parallel_thread_raises_to_ultra(config):
    assert resolve(config, DEV, "독립 작업을 병렬로 진행해줘") == {
        "enabled": True,
        "effort": "ultra",
    }


def test_development_ultra_requires_thread(config):
    assert resolve(config, DEV, "독립 작업을 병렬로 진행해줘", thread_id="") == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_development_mutation_suppresses_ultra(config):
    assert resolve(config, DEV, "운영 변경을 병렬로 진행해줘") == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_reservation_effort_only_ladder(config):
    assert resolve(config, RESERVATION, "오늘 상황을 설명해줘") == {
        "enabled": True,
        "effort": "medium",
    }
    assert resolve(config, RESERVATION, "투숙률 전략 보고서를 작성해줘") == {
        "enabled": True,
        "effort": "high",
    }
    assert resolve(config, RESERVATION, "복합 진단 후 교차 검증해줘") == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_reservation_never_uses_ultra_and_never_returns_model(config):
    result = resolve(config, RESERVATION, "여러 분석을 병렬로 진행해줘")
    assert result == {"enabled": True, "effort": "high"}
    assert "model" not in result


def test_non_discord_and_unbound_channels_do_not_override(config):
    assert resolve_channel_reasoning_config(
        config,
        platform="telegram",
        chat_id=DEV,
        thread_id="thread-1",
        parent_id=DEV,
        message="병렬",
    ) is None
    assert resolve(config, "unbound", "병렬") is None


def test_manual_session_override_has_priority(monkeypatch):
    from gateway.run import GatewayRunner

    state = SimpleNamespace(
        conversation=SimpleNamespace(reasoning_override={"enabled": True, "effort": "low"})
    )
    fake = SimpleNamespace(
        _peek_session_state=lambda key: state,
        _load_reasoning_config=lambda model: {"enabled": True, "effort": "high"},
    )
    resolved = GatewayRunner._resolve_session_reasoning_config(
        fake,
        session_key="session-1",
        model="gpt-5.6-sol",
        channel_reasoning_config={"enabled": True, "effort": "xhigh"},
    )
    assert resolved == {"enabled": True, "effort": "low"}


ESCALATION_ONLY = "1500000000000000001"


@pytest.fixture
def high_floor_config():
    """A channel whose own floor is already above what its terms name.

    `default_effort` accepts every rung of the ladder, so a channel may
    declare `ultra` outright. Term lists then say "at least this much",
    which is only meaningful if a match cannot land below the floor the
    channel already chose.
    """
    return {
        "discord": {
            "channel_reasoning_policies": {
                ESCALATION_ONLY: {
                    "default_effort": "ultra",
                    "high_terms": ["요약"],
                    "xhigh_terms": ["근본 원인"],
                    "ultra": {
                        "enabled": True,
                        "terms": ["병렬"],
                        "requires_thread": True,
                        "blocked_terms": ["배포"],
                    },
                }
            }
        }
    }


def test_high_term_does_not_lower_an_ultra_default(high_floor_config):
    assert resolve(high_floor_config, ESCALATION_ONLY, "요약해줘") == {
        "enabled": True,
        "effort": "ultra",
    }


def test_xhigh_term_does_not_lower_an_ultra_default(high_floor_config):
    assert resolve(high_floor_config, ESCALATION_ONLY, "근본 원인을 찾아줘") == {
        "enabled": True,
        "effort": "ultra",
    }


def test_blocked_ultra_request_does_not_lower_an_ultra_default(high_floor_config):
    # Refusing the escalation is not a reason to drop below the floor the
    # channel configured without any term at all.
    assert resolve(high_floor_config, ESCALATION_ONLY, "병렬로 배포해줘") == {
        "enabled": True,
        "effort": "ultra",
    }


def test_ultra_request_outside_a_thread_does_not_lower_the_default(high_floor_config):
    assert resolve(
        high_floor_config, ESCALATION_ONLY, "병렬로 처리해줘", thread_id=None
    ) == {"enabled": True, "effort": "ultra"}


def test_terms_still_escalate_from_a_low_default(config):
    # The guard above must not turn escalation off: a medium channel still
    # climbs when a term matches.
    assert resolve(config, RESERVATION, "근본 원인 분석") == {
        "enabled": True,
        "effort": "xhigh",
    }


def _turn_runner(session_override=None):
    """A GatewayRunner stand-in carrying only what turn resolution reads."""
    from gateway.run import GatewayRunner

    state = (
        None
        if session_override is None
        else SimpleNamespace(
            conversation=SimpleNamespace(reasoning_override=session_override)
        )
    )
    fake = SimpleNamespace(
        _peek_session_state=lambda key: state,
        _load_reasoning_config=lambda model: {"enabled": True, "effort": "high"},
    )
    fake._resolve_session_reasoning_config = (
        lambda **kwargs: GatewayRunner._resolve_session_reasoning_config(
            fake, **kwargs
        )
    )
    fake._resolve_turn_reasoning_config = (
        lambda **kwargs: GatewayRunner._resolve_turn_reasoning_config(
            fake, **kwargs
        )
    )
    return fake


def _discord_source(chat_id, thread_id="thread-1"):
    return SimpleNamespace(
        platform="discord",
        chat_id=chat_id,
        thread_id=thread_id,
        parent_chat_id=chat_id,
    )


def test_applied_policy_is_reported_when_nothing_outranks_it(monkeypatch, config):
    """Both turn paths log from the second return value, so it must be set."""
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    resolved, applied = _turn_runner()._resolve_turn_reasoning_config(
        source=_discord_source(DEV),
        session_key="session-1",
        model="gpt-5.6-sol",
        message="현재 상태를 확인해줘",
    )
    assert resolved == {"enabled": True, "effort": "high"}
    assert applied == resolved


def test_applied_policy_is_not_reported_when_a_session_override_wins(
    monkeypatch, config
):
    """The turn log must describe the effort the turn actually ran at.

    A session-scoped `/reasoning --session` override outranks any channel
    policy, so reporting the policy merely because it matched would name an
    effort the turn did not use -- defeating what the log exists for.
    """
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    override = {"enabled": True, "effort": "low"}
    resolved, applied = _turn_runner(override)._resolve_turn_reasoning_config(
        source=_discord_source(DEV),
        session_key="session-1",
        model="gpt-5.6-sol",
        message="현재 상태를 확인해줘",
    )
    assert resolved == override
    assert applied is None


def test_policy_values_are_read_after_env_expansion(monkeypatch):
    """`${VAR}` in a policy must be expanded before the effort allowlist.

    Unexpanded, `default_effort` holds the literal placeholder, fails the
    allowlist, and the resolver returns None -- the policy silently does
    nothing instead of reporting a bad value. `_load_gateway_runtime_config`
    exists for exactly this, so turn resolution must read through it.
    """
    from gateway import run as gateway_run

    monkeypatch.setenv("CHANNEL_EFFORT", "xhigh")
    raw = {
        "discord": {
            "channel_reasoning_policies": {
                ESCALATION_ONLY: {"default_effort": "${CHANNEL_EFFORT}"}
            }
        }
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: raw)

    # Read raw, the placeholder survives and the policy resolves to nothing.
    assert resolve(raw, ESCALATION_ONLY, "무엇이든") is None

    resolved, applied = _turn_runner()._resolve_turn_reasoning_config(
        source=_discord_source(ESCALATION_ONLY),
        session_key="session-1",
        model="gpt-5.6-sol",
        message="무엇이든",
    )
    assert resolved == {"enabled": True, "effort": "xhigh"}
    assert applied == resolved
