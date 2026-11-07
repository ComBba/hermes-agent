"""Record abnormally-ended agent turns for the 비서실장 dispatcher.

Cron failures already reach the owner (collect_cron_failures). Interactive turns
did not: a Discord turn that burned its whole iteration budget and stopped looks
exactly like one that finished, because the completion explanation is only
appended when the reply is empty or truncated.

This hook closes that gap on the observation side only. It writes a local
artifact and stays silent otherwise -- per docs/CONDUCTOR_ALERTS_AND_ARTIFACTS.md
individual monitors do not alert the owner directly; the dispatcher decides what
is worth interrupting for.

Deliberately stores no message or response text. The turn's identity, its exit
reason, and how many iterations it burned are enough to triage; the conversation
content is not this hook's business.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

ABNORMAL_REASONS = frozenset({
    "max_iterations_reached",
    "interrupted_by_user",
    "interrupted_during_api_call",
    "budget_exhausted",
    "guardrail_halt",
    "all_retries_exhausted_no_response",
    "empty_response_exhausted",
    "error_near_max_iterations",
    "local_processing_error",
    "ollama_runtime_context_too_small",
    "context_compression_timeout", "context_compression_exhausted",
    "session_persistence_failed", "pending_tool_result", "repeated_outer_errors",
    "runtime_loop_error", "gateway_inactivity_timeout", "gateway_sse_disconnect",
    "gateway_unhandled_exception", "gateway_interrupt_unclassified",
    "gateway_shutdown", "gateway_restart", "gateway_session_evicted", "gateway_stale_generation",
    "interrupted_by_system", "interpreter_shutdown", "review_input_budget_exhausted",
    "redirect_restart_limit_exceeded", "rebuilt_restart_limit_exceeded", "compaction_handoff_not_actionable",
    "api_auth", "api_auth_permanent", "api_rate_limit", "api_upstream_rate_limit",
    "api_server_error", "api_overloaded", "api_timeout", "api_connection_error",
    "api_incomplete_response", "api_invalid_response", "api_ssl_cert_verification", "api_upstream_blocked",
})

# Not every abnormal exit deserves the owner's attention. A user pressing stop
# is a normal thing to do, so it is recorded but marked non-actionable.
NON_ACTIONABLE = frozenset({
    "interrupted_by_user", "interrupted_by_system", "interpreter_shutdown",
    "gateway_shutdown", "gateway_restart", "gateway_session_evicted",
    "gateway_stale_generation", "gateway_interrupt_unclassified",
})
PARAMETERIZED_ERRORS = frozenset({"local_processing_error", "error_near_max_iterations", "repeated_outer_errors"})
ITERATION_REASON = re.compile(r"max_iterations_reached\(\d{1,10}(?:/\d{1,10})?\)\Z")
NORMAL_REASON = re.compile(r"text_response(?:\(finish_reason=(?:stop|length|end_turn|tool_calls|unknown)\))?\Z")

MAX_EVENT_BYTES = 2_000_000


def _runtime_dir() -> Path:
    return Path(
        os.environ.get(
            "HERMES_TURN_EXIT_DIR",
            "/Users/choisungyong/Documents/GitHub/hermes-agent--gemma4-12b-mlx"
            "/logs/runtime/turn-exits",
        )
    )


def _now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _classify(reason: str) -> tuple[bool, bool]:
    """Return (is_abnormal, is_actionable)."""
    normalized = _safe_reason(reason)
    if normalized not in ABNORMAL_REASONS and not ITERATION_REASON.fullmatch(normalized):
        return (False, False)
    return (True, normalized not in NON_ACTIONABLE)


def _safe_reason(reason: str) -> str:
    base, separator, _detail = reason.partition("(")
    if base in PARAMETERIZED_ERRORS and (not separator or reason.endswith(")")):
        return base
    if reason in ABNORMAL_REASONS or ITERATION_REASON.fullmatch(reason):
        return reason
    return "unknown"


def _record_unknown(directory: Path) -> None:
    path = directory / "diagnostics.json"
    count = 0
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        value = previous.get("unknown_reason_count") if isinstance(previous, dict) else None
        if type(value) is int:
            count = max(0, value)
    except (OSError, ValueError):
        count = 0
    path.write_text(json.dumps({
        "unknown_reason_count": min(count + 1, 2_147_483_647), "updated_at": _now_iso(),
    }), encoding="utf-8")


def _trim_ledger(path: Path) -> None:
    """Keep the append-only ledger bounded without losing the recent tail."""
    try:
        if path.stat().st_size <= MAX_EVENT_BYTES:
            return
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[len(lines) // 2:]), encoding="utf-8")
    except OSError:
        pass


async def handle(event_type: str, context: dict[str, Any]) -> None:
    if event_type != "agent:end":
        return

    raw_reason = context.get("turn_exit_reason")
    raw_reason = raw_reason if isinstance(raw_reason, str) else ""
    reason = _safe_reason(raw_reason)
    abnormal, actionable = _classify(reason)
    if not abnormal:
        if not NORMAL_REASON.fullmatch(raw_reason) and raw_reason not in {"partial_stream_recovery", "fallback_prior_turn_content"}:
            try:
                directory = _runtime_dir()
                directory.mkdir(parents=True, exist_ok=True)
                _record_unknown(directory)
            except OSError:
                return
        return

    stale = context.get("stale", False) is not False
    calls = context.get("api_call_count")
    calls = min(max(0, calls), 1_000_000) if type(calls) is int else 0
    kind = context.get("api_call_count_kind", "legacy_iteration")
    if not isinstance(kind, str) or kind not in {"main_chat_client_invocation", "codex_turn_attempt", "legacy_iteration", "unavailable"}:
        kind, calls = "unavailable", 0
    if kind == "unavailable":
        calls = 0

    event = {
        "observed_at": _now_iso(),
        "session_id": str(context.get("session_id") or ""),
        "platform": str(context.get("platform") or ""),
        "chat_id": str(context.get("chat_id") or ""),
        "thread_id": str(context.get("thread_id") or ""),
        "chat_type": str(context.get("chat_type") or ""),
        "user_id": str(context.get("user_id") or ""),
        "turn_exit_reason": reason,
        "api_call_count": calls,
        "api_call_count_kind": kind,
        "stale": stale,
        "actionable": actionable and not stale,
    }

    try:
        directory = _runtime_dir()
        directory.mkdir(parents=True, exist_ok=True)
        ledger = directory / "events.jsonl"
        with ledger.open("a", encoding="utf-8") as handle_:
            handle_.write(json.dumps(event, ensure_ascii=False) + "\n")
        _trim_ledger(ledger)

        (directory / "status.json").write_text(
            json.dumps(
                {
                    "status": "changed" if event["actionable"] else "ok",
                    "sender": "turn-exit-monitor",
                    "reason": f"turn ended as {reason}",
                    "evidence": str(ledger),
                    "next_action": "queued" if event["actionable"] else "observe",
                    "updated_at": event["observed_at"],
                    "latest": event,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        # A monitor that cannot write its artifact must not take the turn down
        # with it. Hook errors are already caught upstream; this keeps the
        # failure from looking like an agent fault in the logs.
        return
