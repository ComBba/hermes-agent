"""Per-channel reasoning-effort policies for gateway turns.

The resolver is deliberately model-agnostic: it may select an effort for a
bound Discord channel/thread, but it never changes the configured model.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping, Optional

from hermes_constants import parse_reasoning_effort


# Ordered weakest to strongest. Terms escalate along this ladder and never
# down it: a channel that configures ``default_effort: ultra`` has already
# decided its floor, and a message that happens to contain a ``high_terms``
# entry is asking for at least high, not for exactly high.
_EFFORT_LADDER = ("medium", "high", "xhigh", "ultra")
_ALLOWED_EFFORTS = frozenset(_EFFORT_LADDER)


def _at_least(effort: str, floor: str) -> str:
    return max(effort, floor, key=_EFFORT_LADDER.index)


def _platform_name(platform: Any) -> str:
    value = getattr(platform, "value", platform)
    return str(value or "").strip().casefold()


def _normalize(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(term for raw in value if (term := _normalize(raw).strip()))


def _contains_any(text: str, raw_terms: Any) -> bool:
    return any(term in text for term in _terms(raw_terms))


def _policy_for_ids(policies: Mapping[str, Any], *ids: Any) -> Optional[Mapping[str, Any]]:
    for raw_id in ids:
        channel_id = str(raw_id or "").strip()
        if not channel_id:
            continue
        policy = policies.get(channel_id)
        if isinstance(policy, Mapping):
            return policy
    return None


def _parsed_effort(effort: Any) -> Optional[dict]:
    normalized = _normalize(effort).strip()
    if normalized not in _ALLOWED_EFFORTS:
        return None
    return parse_reasoning_effort(normalized)


def resolve_channel_reasoning_config(
    user_config: Mapping[str, Any] | None,
    *,
    platform: Any,
    chat_id: Any,
    thread_id: Any = None,
    parent_id: Any = None,
    message: Any = "",
) -> Optional[dict]:
    """Return a model-agnostic reasoning override for a bound Discord turn.

    Resolution is deterministic and config-only. A thread may inherit its
    parent channel's policy. Explicit parallel escalation is allowed only when
    the policy enables it, the request is in a thread when required, and no
    configured blocked term appears.
    """

    if _platform_name(platform) != "discord" or not isinstance(user_config, Mapping):
        return None

    discord_cfg = user_config.get("discord")
    if not isinstance(discord_cfg, Mapping):
        return None
    policies = discord_cfg.get("channel_reasoning_policies")
    if not isinstance(policies, Mapping):
        return None

    policy = _policy_for_ids(policies, thread_id, chat_id, parent_id)
    if policy is None:
        return None

    default = _normalize(policy.get("default_effort")).strip()
    if default not in _ALLOWED_EFFORTS:
        return None

    text = _normalize(message)
    effort = default
    if _contains_any(text, policy.get("high_terms")):
        effort = _at_least(effort, "high")
    if _contains_any(text, policy.get("xhigh_terms")):
        effort = _at_least(effort, "xhigh")

    ultra_cfg = policy.get("ultra")
    if isinstance(ultra_cfg, Mapping) and ultra_cfg.get("enabled") is True:
        parallel_requested = _contains_any(text, ultra_cfg.get("terms"))
        blocked = _contains_any(text, ultra_cfg.get("blocked_terms"))
        requires_thread = ultra_cfg.get("requires_thread") is not False
        in_thread = bool(str(thread_id or "").strip())
        if parallel_requested:
            if blocked or (requires_thread and not in_thread):
                # Refusing the escalation is not a reason to drop below the
                # channel's own floor: a blocked ultra request in a channel
                # configured for ultra still gets what the channel configured.
                effort = _at_least(effort, "xhigh")
            else:
                effort = "ultra"

    return _parsed_effort(effort)
