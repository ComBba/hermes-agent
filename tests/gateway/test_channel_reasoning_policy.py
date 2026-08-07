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
