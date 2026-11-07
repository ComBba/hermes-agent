"""Final-reply boundary for the vision.ocr_command opt-in (gemma-operator, 2026-10-19).

``agent.image_routing`` records an independent local OCR reading (or its failure) for each image,
with an id that also appears in the image hint of the user message. ``pre_llm_call`` (once per turn)
binds to its ``turn_id`` only the checks whose id is in that turn's user message and drops the rest,
so an interrupted image turn never labels a later turn. Checks made by a tool call during the turn
(vision_analyze) are recorded under ``turn:<turn_id>`` and merged by the final hook for that turn only. ``transform_llm_output`` runs before the
assistant row is first persisted, so the framed reply is what Discord shows and what the history
replays. Scope: non-streaming Discord only; other platforms are left unchanged. The frame never says
a value is correct: OCR is a second fallible reading, and agreement is not proof.

Only the first string a ``transform_llm_output`` hook returns is used. When another output
transform is registered, this hook returns ``None`` instead of framing the reply: the host alone
owns callback invocation, signature adaptation, and result selection. Image-turn tool blocking
still applies; the OCR warning frame is only available when this is the sole output transform.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[,_./:-][A-Za-z0-9]+)*")
_SHORT_NUMBER_LABEL = re.compile(r"(?:재고|수량|금액|가격|날짜|일자|객실|방|코드)\s*[:#]?\s*$", re.IGNORECASE)
_SEPARATORS = "-,_./:"
_CHECK_ID = re.compile(r"\bocr-[0-9a-f]{8}\b")
_TTL_SECONDS = 1800
_BOUND: dict[str, tuple[float, list[dict[str, Any]]]] = {}  # turn_id -> (bound at, checks)
_IMAGE_TURNS: dict[str, float] = {}
_LOCK = threading.Lock()
_BLOCK_MESSAGE = (
    "이미지 판독값만으로 자동 업무를 실행할 수 없습니다. "
    "원본 구조화 데이터를 확인한 뒤 다시 실행하세요."
)


def _identifiers(text: str) -> list[str]:
    seen: list[str] = []
    for match in _IDENTIFIER.finditer(text or ""):
        token = match.group()
        has_digit = any(c.isdigit() for c in token)
        is_short_number = token.isdigit() and len(token) < 4
        labelled = bool(_SHORT_NUMBER_LABEL.search((text or "")[:match.start()]))
        if has_digit and (not is_short_number or labelled) and token not in seen:
            seen.append(token)
    return seen


def _shape(token: str) -> tuple[int, tuple[int, ...]]:
    return len(token), tuple(i for i, c in enumerate(token) if c in _SEPARATORS)


def compare(reply: str, ocr_lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """``(disagreements, unmatched, agreed)`` over the identifiers in *reply*: same-shape OCR tokens
    that may differ (``[reply/ocr]`` per differing character), identifiers OCR has nothing for, and
    identifiers OCR reads the same way. Flattened OCR cannot establish which tokens share an item;
    an exact reading must not hide another same-shape candidate."""
    ocr_tokens = _identifiers(" ".join(ocr_lines))
    disagreements: list[str] = []
    unmatched: list[str] = []
    agreed: list[str] = []
    for token in _identifiers(reply):
        same_shape = [o for o in ocr_tokens if _shape(o) == _shape(token)]
        alternatives = [o for o in same_shape if o != token]
        if token in same_shape:
            agreed.append(token)
        if not same_shape:
            unmatched.append(token)
        elif alternatives:
            other = min(alternatives, key=lambda o: sum(a != b for a, b in zip(token, o)))
            disagreements.append("".join(a if a == b else f"[{a}/{b}]" for a, b in zip(token, other)))
    return disagreements, unmatched, agreed


def _code(text: str) -> str:
    """Inline code span: Discord renders no markdown or mentions inside it."""
    return "`" + str(text).replace("`", "ˋ").replace("\n", " ") + "`"


def frame(reply: str, checks: list[dict[str, Any]]) -> str:
    lines = [
        (
            "⚠️ **이미지 판독 미검증** — 아래 모델 답변은 판독 초안입니다. "
            "이미지에서 읽은 코드·숫자·날짜·금액을 확정값이나 자동 업무 근거로 쓰지 마세요."
        )
    ]
    for check in checks:
        if check.get("status") != "read":
            lines.append(f"- 독립 OCR 확인 불가({_code(check.get('reason') or 'unknown')}): 비교 근거가 없어 미검증입니다.")
            continue
        ocr_lines = check.get("lines") or []
        lines.append(f"- 독립 OCR {len(ocr_lines)}개 후보 확보(비신뢰 데이터, 원문 비표시).")
        disagreements, unmatched, agreed = compare(reply, ocr_lines)
        if disagreements:
            lines.append("- 같은 형태의 다른 OCR 후보 [모델/OCR]: " + ", ".join(_code(d) for d in disagreements)
                         + " — 같은 항목인지 미확인, 값을 확정하지 않습니다.")
        if unmatched:
            lines.append("- OCR에 대응 판독이 없는 식별자: " + ", ".join(_code(u) for u in unmatched) + " — 미검증.")
        if agreed and not disagreements and not unmatched:
            lines.append("- 모델 판독과 OCR이 같은 식별자: " + ", ".join(_code(a) for a in agreed)
                         + " — 같아도 정확성을 보장하지 않습니다.")
        if not (disagreements or unmatched or agreed):
            lines.append("- 답변에서 비교할 식별자를 찾지 못했습니다 — 비교 대상 없음, 미검증.")
    lines += ["", "**[모델 판독 초안 — 미검증]**", reply]
    return "\n".join(lines)


def _discord_streaming() -> bool:
    try:
        from hermes_cli.config import load_config

        display = (load_config().get("display") or {}).get("platforms") or {}
        return bool((display.get("discord") or {}).get("streaming"))
    except Exception:  # config boundary fails safe by leaving output untouched
        return True  # unknown -> treat as streaming and leave the reply untouched


def _expire(now: float) -> None:
    for turn_id in [t for t, (at, _) in _BOUND.items() if now - at > _TTL_SECONDS]:
        del _BOUND[turn_id]
    for turn_id in [t for t, at in _IMAGE_TURNS.items() if now - at > _TTL_SECONDS]:
        del _IMAGE_TURNS[turn_id]


def _remember_image_turn(turn_id: str, checks: list[dict[str, Any]] | None = None) -> None:
    if not turn_id:
        return
    now = time.monotonic()
    with _LOCK:
        _expire(now)
        _IMAGE_TURNS[turn_id] = now
        if checks:
            existing = _BOUND.get(turn_id, (now, []))[1]
            seen = {str(check.get("id") or "") for check in existing}
            existing += [check for check in checks if str(check.get("id") or "") not in seen]
            _BOUND[turn_id] = (now, existing)


def _collect_tool_checks(turn_id: str) -> None:
    if not turn_id:
        return
    try:
        from agent.image_routing import pop_ocr_checks

        checks = pop_ocr_checks(f"turn:{turn_id}")
    except Exception as exc:  # optional OCR bridge must not break tool dispatch
        logger.warning("image-ocr-guard: could not read tool-call OCR checks: %s", exc)
        return
    if checks:
        _remember_image_turn(turn_id, checks)


def _current_user_text(user_message: Any, conversation_history: Any) -> str:
    """The turn's user message as the model gets it. The gateway passes the persisted text (without
    the image hints) as ``user_message``; the API form with the hints is the last user entry of
    ``conversation_history`` (the hook receives the turn's messages, current user turn included)."""
    last_user = next((m for m in reversed(conversation_history or []) if isinstance(m, dict)
                      and m.get("role") == "user"), None)
    return f"{user_message}\n{(last_user or {}).get('content')}"


def pre_llm_call(turn_id: str = "", user_message: Any = None, conversation_history: Any = None,
                 **_: Any) -> None:
    """Turn start: bind this turn's own OCR checks to *turn_id*; drop leftovers of earlier turns."""
    try:
        from agent.image_routing import pop_ocr_checks
        from tools.approval_context import get_current_session_key

        recorded = pop_ocr_checks(get_current_session_key(""))
    except Exception as exc:  # optional OCR bridge must not break turn setup
        logger.warning("image-ocr-guard: could not read OCR checks: %s", exc)
        return
    ids = set(_CHECK_ID.findall(_current_user_text(user_message, conversation_history)))
    mine = [check for check in recorded if check.get("id") in ids]
    if len(mine) != len(recorded):
        logger.info("image-ocr-guard: dropped %d OCR check(s) not in this turn", len(recorded) - len(mine))
    now = time.monotonic()
    with _LOCK:
        _expire(now)
        if mine and turn_id:
            _BOUND[str(turn_id)] = (now, mine)
            _IMAGE_TURNS[str(turn_id)] = now
    return


def pre_tool_call(tool_name: str = "", turn_id: str = "", **_: Any) -> dict[str, str] | None:
    """Allow inspection, but block effects while an image-derived value is in the active turn."""
    turn_id = str(turn_id or "")
    if not turn_id:
        return None
    if tool_name == "vision_analyze":
        _remember_image_turn(turn_id)
    _collect_tool_checks(turn_id)
    with _LOCK:
        _expire(time.monotonic())
        image_turn = turn_id in _IMAGE_TURNS
    if not image_turn:
        return None
    try:
        from agent.tool_result_classification import tool_may_have_side_effect
    except (ImportError, AttributeError):
        return {"action": "block", "message": _BLOCK_MESSAGE}
    if not tool_may_have_side_effect(str(tool_name or "")):
        return None
    return {"action": "block", "message": _BLOCK_MESSAGE}


def _other_transforms() -> list[Callable[..., Any]]:
    """The other ``transform_llm_output`` callbacks the plugin manager has loaded, in its order."""
    from hermes_cli.plugins import get_plugin_manager

    return [cb for cb in get_plugin_manager()._hooks.get("transform_llm_output", [])
            if getattr(cb, "__module__", None) != __name__]


def transform_llm_output(response_text: str = "", platform: str = "", turn_id: str = "", **_: Any) -> str | None:
    _collect_tool_checks(str(turn_id or ""))
    with _LOCK:
        _expire(time.monotonic())
        _at, checks = _BOUND.pop(str(turn_id), (0.0, []))
        _IMAGE_TURNS.pop(str(turn_id), None)
    if not checks or not response_text:
        return None
    if str(platform).lower() != "discord" or _discord_streaming():
        logger.info("image-ocr-guard: %d OCR check(s) not framed (platform=%s, outside scope)", len(checks), platform)
        return None
    try:
        if _other_transforms():
            logger.info("image-ocr-guard: OCR warning frame deferred to other output transforms")
            return None
    except Exception as exc:  # host plugin enumeration is an isolation boundary
        logger.warning("image-ocr-guard: could not list other transforms: %s", exc)
        return None
    return frame(response_text, checks)


def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("transform_llm_output", transform_llm_output)
