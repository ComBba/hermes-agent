from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Final, assert_never

from .models import ProbeReport, State, digest, report_to_json
from .task_models import CompletionDraft, CompletionState, Finding, Inspection

_FINDINGS: Final = {
    Finding.HEALTH_PARSE_FAILED: "원 실행에서 건강 응답의 JSON 해석이 실패했습니다.",
    Finding.MIGRATION_HTML: "원 실행의 응답은 주소 변경 HTML이었습니다.",
    Finding.RECOVERY_NOT_OBSERVED: "확인한 원 실행 구간에서 현재 대상 재탐색을 실행한 기록이 없습니다.",
}


def render(state: CompletionState, evidence_sha256: str, findings: tuple[Finding, ...] = ()) -> CompletionDraft:
    retains_pending = True
    match state:
        case CompletionState.HEALTH_VERIFIED:
            message = "현재 배포 근거에서 확인한 대상의 서비스 식별·건강·데이터 최신성이 검증됐습니다."
            retains_pending = False
        case CompletionState.AUTH_BLOCKED:
            message = "현재 대상 확인은 인증 제한으로 차단됐습니다. 접근 제한이며 서비스 장애나 정상 작동의 증거는 아닙니다."
        case CompletionState.STATUS_BLOCKED:
            message = "현재 상태 확인에 필요한 배포 근거·서비스 식별 또는 최신성을 검증하지 못했습니다. 확인 요청을 미완료로 유지합니다."
        case CompletionState.INSPECTED:
            message = "원 문의의 실제 입력과 실행 기록을 조사했고 현재 대상의 상태도 검증했습니다. 모델 내부 원인은 확정하지 않습니다."
            retains_pending = False
        case CompletionState.EVIDENCE_UNLINKED:
            message = "조사 요청을 선택했지만 동일 범위의 원 문의·실행 증거를 연결하지 못했습니다. 원 문의에 답글로 연결하거나 취소할 수 있습니다."
        case CompletionState.INSPECTION_BLOCKED:
            message = "원 문의의 실행 증거를 다시 검증하지 못했습니다. 설명만으로 조사를 완료하지 않고 요청을 유지합니다."
        case CompletionState.INPUT_BLOCKED:
            message = "관리형 문의 입력을 처리할 수 없습니다. 상태 확인, 조사 또는 /inquiry cancel을 명시해 주세요."
        case CompletionState.PENDING_TASK_BLOCKED:
            message = "아직 해결되지 않은 문의가 있습니다. 후속 질문으로 계속 확인하거나 /inquiry cancel로 해당 문의를 취소해 주세요."
        case CompletionState.CANCELLED:
            message = "이 세션 범위의 문의를 취소했습니다."
            retains_pending = False
        case CompletionState.TIMED_OUT:
            message = "허용된 확인 시간이 초과됐습니다. 결과를 추정하지 않고 문의를 미완료로 유지합니다."
        case CompletionState.HANDLER_ERROR:
            message = "관리형 확인 경로에서 오류가 발생했습니다. 일반 모델 답변으로 대체하지 않고 문의를 미완료로 유지합니다."
        case unreachable:
            assert_never(unreachable)
    details = "\n".join(_FINDINGS[item] for item in findings)
    return CompletionDraft(state, message + ("\n" + details if details else ""), retains_pending, evidence_sha256)


def render_probe(report: ProbeReport, inspection: Inspection | None = None) -> CompletionDraft:
    state = CompletionState.STATUS_BLOCKED
    if report.state == State.HEALTHY:
        state = CompletionState.INSPECTED if inspection is not None else CompletionState.HEALTH_VERIFIED
    if report.state == State.AUTH_BLOCKED:
        state = CompletionState.AUTH_BLOCKED
    evidence = digest((report_to_json(report) + (json.dumps(asdict(inspection), sort_keys=True) if inspection is not None else "")).encode())
    draft = render(state, evidence, inspection.findings if inspection is not None else ())
    labels = {
        State.IDENTITY_UNVERIFIED: "서비스 식별 계약이 확인되지 않았습니다.",
        State.IDENTITY_MISMATCH: "응답의 서비스 식별이 배포 계약과 일치하지 않습니다.",
        State.FRESHNESS_UNVERIFIED: "응답 데이터의 최신성을 검증할 수 없습니다.",
        State.DATA_STALE: "응답 데이터가 허용된 최신성 범위를 벗어났습니다.",
        State.DISCOVERY_CONFLICT: "현재 배포 근거가 서로 충돌합니다.",
        State.DISCOVERY_PROCESS_MISMATCH: "실행 프로세스와 배포 정의가 일치하지 않습니다.",
        State.DISCOVERY_STALE: "배포 근거의 관측 시점이 유효 범위를 벗어났습니다.",
        State.REDISCOVERY_EXHAUSTED: "승인된 배포 근거로 재탐색했지만 호환되는 건강 응답을 확인하지 못했습니다.",
    }
    explanation = labels.get(report.state)
    return replace(draft, text=draft.text + ("\n" + explanation if explanation else ""), detail_code=report.state.value)
