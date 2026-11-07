"""REL-01 hook metadata with a closed reason taxonomy and bounded call count.

Unknown reasons remain unknown; provider/exception detail never crosses this
artifact boundary. System interrupts remain distinct from user stops. Stale
output is observed but must not cause delivery or an actionable monitor alert.
"""

from __future__ import annotations

import re
from typing import Any, Final, Literal, TypedDict

REASON_MAX_CHARS = 200
GATEWAY_UNHANDLED_EXCEPTION = "gateway_unhandled_exception"
GATEWAY_STALE_GENERATION = "gateway_stale_generation"
_EXACT_REASONS: Final = frozenset({
    "unknown", "text_response", "interrupted_by_user", "interrupted_by_system",
    "interrupted_during_api_call", "budget_exhausted", "guardrail_halt",
    "all_retries_exhausted_no_response", "empty_response_exhausted",
    "ollama_runtime_context_too_small", "context_compression_timeout",
    "context_compression_exhausted", "session_persistence_failed", "pending_tool_result",
    "interpreter_shutdown", "review_input_budget_exhausted", "redirect_restart_limit_exceeded",
    "rebuilt_restart_limit_exceeded", "compaction_handoff_not_actionable",
    "partial_stream_recovery", "fallback_prior_turn_content",
    "gateway_inactivity_timeout", "gateway_sse_disconnect", "gateway_shutdown",
    "gateway_restart", "gateway_session_evicted", "gateway_stale_generation",
    "gateway_unhandled_exception", "gateway_interrupt_unclassified",
})
_DETAIL_REASONS: Final = frozenset({"local_processing_error", "repeated_outer_errors", "error_near_max_iterations"})
_ITERATION_REASON: Final = re.compile(r"max_iterations_reached\(\d{1,10}(?:/\d{1,10})?\)\Z")
_TEXT_REASON: Final = re.compile(r"text_response\(finish_reason=(?:stop|length|end_turn|tool_calls|unknown)\)\Z")
_FAILURE_REASONS: Final = {
    "auth": "api_auth", "auth_permanent": "api_auth_permanent",
    "rate_limit": "api_rate_limit", "upstream_rate_limit": "api_upstream_rate_limit",
    "server_error": "api_server_error", "overloaded": "api_overloaded", "timeout": "api_timeout",
    "gateway_connection": "api_connection_error", "incomplete_response": "api_incomplete_response",
    "invalid_response": "api_invalid_response", "ssl_cert_verification": "api_ssl_cert_verification",
    "upstream_blocked": "api_upstream_blocked", "empty_response": "empty_response_exhausted",
    "loop_error": "runtime_loop_error",
    "truncated": "api_incomplete_response",
}


def _normalize_interrupt_message(message: object) -> str:
    """Normalize an interrupt marker without trusting its input type."""
    return " ".join(message.strip().split()).lower() if isinstance(message, str) else ""


def _normalize_reason(reason: object) -> str:
    """Retain known codes and numeric budgets, discarding provider/exception detail."""
    if not isinstance(reason, str):
        return "unknown"
    if reason in _EXACT_REASONS or reason in _FAILURE_REASONS.values() or _ITERATION_REASON.fullmatch(reason) or _TEXT_REASON.fullmatch(reason):
        return reason
    base, separator, _detail = reason.partition("(")
    if base in _DETAIL_REASONS and (not separator or reason.endswith(")")):
        return base
    if base == "interrupted_by_system" and reason.endswith(")"):
        return base
    return "unknown"


def _interrupt_tables() -> tuple[dict[str, str], frozenset[str]]:
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
    if not isinstance(agent_result, dict):
        return "unknown"
    reason = _normalize_reason(agent_result.get("turn_exit_reason"))
    if reason != "unknown" and not reason.startswith("interrupt"):
        return reason
    if agent_result.get("interrupted") is not True:
        failure = agent_result.get("failure_reason")
        if reason == "unknown" and failure == "context_overflow" and agent_result.get("compression_exhausted") is True:
            return "context_compression_exhausted"
        return _FAILURE_REASONS.get(failure, "unknown") if isinstance(failure, str) and reason == "unknown" else reason
    message = _normalize_interrupt_message(agent_result.get("interrupt_message"))
    system, user = _interrupt_tables()
    if message in system:
        return system[message]
    if message in user:
        return "interrupted_by_user"
    if reason != "unknown":
        return reason
    from gateway import run as gateway_run

    if message and not gateway_run._is_control_interrupt_message(message):
        return "interrupted_by_user"
    return "gateway_interrupt_unclassified"


class ApiCallObservation(TypedDict):
    kind: Literal["main_chat_client_invocation", "codex_turn_attempt"]
    count: int


def validated_api_call_observation(value: Any) -> ApiCallObservation | None:
    if (type(value) is not dict or len(value) != 2
            or any(type(key) is not str for key in value) or set(value) != {"kind", "count"}):
        return None
    kind, count = value["kind"], value["count"]
    if type(count) is not int or not 0 <= count <= 1_000_000:
        return None
    if type(kind) is not str:
        return None
    if kind == "main_chat_client_invocation":
        return {"kind": "main_chat_client_invocation", "count": count}
    if kind == "codex_turn_attempt":
        return {"kind": "codex_turn_attempt", "count": count}
    return None


def agent_end_metadata(agent_result: Any, *, stale: bool = False) -> dict[str, Any]:
    """Bounded, fail-safe ``agent:end`` termination metadata."""
    count = agent_result.get("api_calls") if isinstance(agent_result, dict) else None
    if isinstance(count, str) and count.isascii() and count.isdigit() and len(count) <= 7:
        count = int(count)
    api_call_count = min(max(0, count), 1_000_000) if type(count) is int else 0
    kind = "legacy_iteration"
    if isinstance(agent_result, dict) and "api_call_observation" in agent_result:
        observation = validated_api_call_observation(agent_result["api_call_observation"])
        if observation is not None:
            kind = observation["kind"]
            api_call_count = observation["count"]
        else:
            kind, api_call_count = "unavailable", 0
    reason = turn_exit_reason(agent_result)
    if stale and (reason == "unknown" or reason.startswith("text_response")):
        # A superseded run that otherwise finished normally: say why its output is dropped.
        reason = GATEWAY_STALE_GENERATION
    return {
        "turn_exit_reason": reason[:REASON_MAX_CHARS],
        "api_call_count": api_call_count,
        "api_call_count_kind": kind,
        "stale": bool(stale),
    }


def crash_metadata() -> dict[str, Any]:
    """Metadata for a turn whose body raised before any ``agent:end`` was emitted."""
    return {"turn_exit_reason": GATEWAY_UNHANDLED_EXCEPTION, "api_call_count": 0,
            "api_call_count_kind": "unavailable", "stale": False}
