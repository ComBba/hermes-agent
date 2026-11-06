"""Termination metadata for the ``agent:end`` hook (REL-01).

The ``agent:end`` context carried only the response text, so a hook could not
tell a turn that finished normally from one that hit its iteration budget, was
interrupted, halted on a guardrail, or crashed. The response text cannot carry
that distinction: an explanation is appended only when the reply is empty or
truncated, so a substantive summary emitted at the iteration limit is
indistinguishable from a clean finish downstream.

``agent_end_metadata`` adds three bounded, fail-safe fields:

- ``turn_exit_reason`` -- why the turn ended, at most 200 characters, collapsed
  to a single printable line. Specific agent-finalizer reasons pass through.
  Gateway-owned classes are ``interrupted_by_user``, ``unknown``, and the
  ``gateway_*`` family (system aborts stay distinct from user stops).
- ``api_call_count`` -- non-negative API iterations; malformed or absent counts
  clamp to zero, so zero can also mean "not reported".
- ``stale`` -- true when a newer run generation superseded the turn and its
  output is being discarded. Delivery-side handlers must gate on ``stale``,
  never on a reason string.

Every helper swallows its own failures: metadata must never stop the hook from
being emitted, and must never raise into the turn.
"""

from __future__ import annotations

from typing import Any, Dict

REASON_MAX_CHARS = 200
GATEWAY_UNHANDLED_EXCEPTION = "gateway_unhandled_exception"
GATEWAY_STALE_GENERATION = "gateway_stale_generation"


def _normalize_interrupt_message(message: object) -> str:
    """Normalize an interrupt marker without trusting its input type."""
    try:
        if not message:
            return ""
        return " ".join(str(message).strip().split()).lower()
    except Exception:
        return ""


def _normalize_reason(reason: object) -> str:
    """Collapse untrusted finalizer reason text to a single printable line."""
    try:
        if reason is None:
            return ""
        collapsed = " ".join(str(reason).strip().split())
        return "".join(char for char in collapsed if char.isprintable())
    except Exception:
        return ""


def _interrupt_tables() -> tuple[Dict[str, str], frozenset]:
    """System markers -> gateway reason, and the user-stop markers.

    Read lazily from ``gateway.run`` so the markers stay single-sourced there.
    """
    from gateway import run as gateway_run

    system = {
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_TIMEOUT): "gateway_inactivity_timeout",
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_SSE_DISCONNECT): "gateway_sse_disconnect",
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_GATEWAY_SHUTDOWN): "gateway_shutdown",
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_GATEWAY_RESTART): "gateway_restart",
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_EVICTED): "gateway_session_evicted",
    }
    user = frozenset({
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_STOP),
        _normalize_interrupt_message(gateway_run._INTERRUPT_REASON_RESET),
    })
    return system, user


def turn_exit_reason(agent_result: Any) -> str:
    """A stable reason for the ``agent:end`` hook.

    Normal results carry the finalizer's explicit reason. Early interrupts can
    bypass the finalizer, so classify their gateway control marker instead of
    collapsing system aborts into user stops. A non-control interrupt message is
    user-originated correction text; only an interrupted turn with no marker
    stays unclassified.
    """
    try:
        if not isinstance(agent_result, dict):
            return "unknown"
        reason = _normalize_reason(agent_result.get("turn_exit_reason"))
        if reason and not reason.casefold().startswith("interrupt"):
            return reason
        try:
            interrupted = bool(agent_result.get("interrupted"))
        except Exception:
            interrupted = False
        if not interrupted:
            return reason or "unknown"
        message = _normalize_interrupt_message(agent_result.get("interrupt_message"))
        system, user = _interrupt_tables()
        if message in system:
            return system[message]
        if message in user:
            return "interrupted_by_user"
        if reason:
            return reason
        from gateway import run as gateway_run

        if message and not gateway_run._is_control_interrupt_message(message):
            return "interrupted_by_user"
        return "gateway_interrupt_unclassified"
    except Exception:
        return "unknown"


def agent_end_metadata(agent_result: Any, *, stale: bool = False) -> Dict[str, Any]:
    """Bounded, fail-safe ``agent:end`` termination metadata."""
    try:
        api_call_count = max(0, int(agent_result.get("api_calls") or 0))
    except Exception:
        api_call_count = 0
    reason = turn_exit_reason(agent_result)
    if stale and (reason == "unknown" or reason.startswith("text_response")):
        # A superseded run that otherwise finished normally: say why its output is dropped.
        reason = GATEWAY_STALE_GENERATION
    return {
        "turn_exit_reason": reason[:REASON_MAX_CHARS],
        "api_call_count": api_call_count,
        "stale": bool(stale),
    }


def crash_metadata() -> Dict[str, Any]:
    """Metadata for a turn whose body raised before any ``agent:end`` was emitted."""
    return {"turn_exit_reason": GATEWAY_UNHANDLED_EXCEPTION, "api_call_count": 0, "stale": False}
