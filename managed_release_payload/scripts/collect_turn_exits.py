#!/usr/bin/env python3
"""Adapt turn-exit hook artifacts into conductor alert candidates.

The hook owns observation and persistence. This module is the read-only
conductor boundary: it accepts only recent, recognized, actionable records and
returns deterministic alert dictionaries. It does not send messages, retry
work, restart services, or mutate the source ledger.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

ACTIONABLE_EXACT_REASONS = {
    "interrupted_during_api_call",
    "budget_exhausted",
    "guardrail_halt",
    "all_retries_exhausted_no_response",
    "empty_response_exhausted",
    "ollama_runtime_context_too_small",
    "local_processing_error", "error_near_max_iterations", "repeated_outer_errors",
    "context_compression_timeout", "context_compression_exhausted",
    "session_persistence_failed", "pending_tool_result", "runtime_loop_error",
    "gateway_inactivity_timeout", "gateway_sse_disconnect", "gateway_unhandled_exception",
    "review_input_budget_exhausted", "redirect_restart_limit_exceeded",
    "rebuilt_restart_limit_exceeded", "compaction_handoff_not_actionable",
    "api_auth", "api_auth_permanent", "api_rate_limit", "api_upstream_rate_limit",
    "api_server_error", "api_overloaded", "api_timeout", "api_connection_error",
    "api_incomplete_response", "api_invalid_response", "api_ssl_cert_verification", "api_upstream_blocked",
}
ACTIONABLE_PARAMETERIZED_PREFIXES = (
    "max_iterations_reached(",
    "error_near_max_iterations(",
    "local_processing_error(",
    "repeated_outer_errors(",
)
ITERATION_REASON = re.compile(r"max_iterations_reached\(\d{1,10}(?:/\d{1,10})?\)\Z")
DEFAULT_LOOKBACK_HOURS = 6
DEFAULT_MAX_EVENTS = 5


def _parse_observed_at(value: Any) -> dt.datetime | None:
    try:
        observed = dt.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if observed.tzinfo is None:
        return None
    return observed


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _event_key(event: dict[str, Any]) -> str:
    return "turnexit:{session}:{reason}:{observed}".format(
        session=str(event.get("session_id") or "?"),
        reason=_safe_reason(event.get("turn_exit_reason")),
        observed=str(event.get("observed_at") or ""),
    )


def _to_alert(event: dict[str, Any], ledger: Path) -> dict[str, str]:
    reason = _safe_reason(event.get("turn_exit_reason"))
    session = str(event.get("session_id") or "?")
    platform = str(event.get("platform") or "?")
    calls = _safe_int(event.get("api_call_count"))
    kind = event.get("api_call_count_kind", "legacy_iteration")
    if not isinstance(kind, str) or kind not in {"main_chat_client_invocation", "codex_turn_attempt", "legacy_iteration", "unavailable"}:
        kind, calls = "unavailable", 0
    if kind == "unavailable":
        calls = 0
    limit_hit = reason.startswith("max_iterations_reached")
    return {
        "key": _event_key(event),
        "owner": "conductor",
        "title": f"Abnormal agent turn: {platform} ({reason.split('(')[0]})",
        "why": (
            "The turn exhausted its iteration budget and may have returned a "
            "plausible but incomplete response."
            if limit_hit
            else "The turn ended abnormally and requires outcome verification."
        ),
        "evidence": (
            f"session={session}, platform={platform}, reason={reason}, "
            f"api_calls={calls}, count_kind={kind}"
        ),
        "artifact": str(ledger),
        "next_step": (
            "Verify the requested outcome and continue the same session if incomplete."
            if limit_hit
            else "Inspect the final turn outcome and retry only when safe."
        ),
        "url": "",
    }


def _safe_reason(reason: Any) -> str:
    if not isinstance(reason, str):
        return "unknown"
    if reason in ACTIONABLE_EXACT_REASONS or ITERATION_REASON.fullmatch(reason):
        return reason
    if reason.endswith(")") and reason.startswith(ACTIONABLE_PARAMETERIZED_PREFIXES[1:]):
        return reason.partition("(")[0]
    return "unknown"


def collect_turn_exit_alerts(
    ledger: Path,
    *,
    observed_now: dt.datetime | None = None,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> list[dict[str, str]]:
    """Return recent actionable turn exits without mutating ``ledger``."""
    if max_events <= 0 or not ledger.is_file():
        return []

    current = observed_now or dt.datetime.now().astimezone()
    if current.tzinfo is None:
        raise ValueError("observed_now must be timezone-aware")
    cutoff = current - dt.timedelta(hours=max(0, lookback_hours))

    accepted: list[tuple[dt.datetime, dict[str, Any]]] = []
    try:
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if (
                    not isinstance(event, dict) or event.get("actionable") is not True
                    or event.get("stale", False) is not False
                ):
                    continue
                reason = _safe_reason(event.get("turn_exit_reason"))
                if reason == "unknown":
                    continue
                observed = _parse_observed_at(event.get("observed_at"))
                if observed is None or observed < cutoff or observed > current:
                    continue
                accepted.append((observed, event))
    except OSError:
        return []

    alerts: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, event in sorted(accepted, key=lambda item: item[0], reverse=True):
        key = _event_key(event)
        if key in seen:
            continue
        seen.add(key)
        alerts.append(_to_alert(event, ledger))
        if len(alerts) >= max(0, max_events):
            break
    return alerts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect actionable Hermes turn-exit alerts as JSON."
    )
    parser.add_argument("ledger", type=Path, help="turn-exit events.jsonl path")
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    alerts = collect_turn_exit_alerts(
        args.ledger,
        lookback_hours=args.lookback_hours,
        max_events=args.max_events,
    )
    print(json.dumps(alerts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
