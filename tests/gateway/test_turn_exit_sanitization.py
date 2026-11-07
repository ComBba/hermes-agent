import pytest

from gateway.turn_exit_metadata import agent_end_metadata


@pytest.mark.parametrize("reason,expected", [
    ("local_processing_error(token-secret)", "local_processing_error"),
    ("repeated_outer_errors(private-exception)", "repeated_outer_errors"),
    ("error_near_max_iterations(secret-key)", "error_near_max_iterations"),
    ("new_provider_reason(private-data)", "unknown"),
    ("max_iterations_reached(12/90)", "max_iterations_reached(12/90)"),
    ("max_iterations_reached(secret)", "unknown"),
])
def test_only_typed_reason_and_numeric_iteration_detail_leave_the_gateway(reason, expected):
    assert agent_end_metadata({"turn_exit_reason": reason})["turn_exit_reason"] == expected


@pytest.mark.parametrize("failure,expected", [
    ("auth", "api_auth"), ("auth_permanent", "api_auth_permanent"),
    ("rate_limit", "api_rate_limit"), ("server_error", "api_server_error"),
    ("timeout", "api_timeout"), ("gateway_connection", "api_connection_error"),
    ("incomplete_response", "api_incomplete_response"), ("invalid_response", "api_invalid_response"),
    ("empty_response", "empty_response_exhausted"), ("loop_error", "runtime_loop_error"),
    ("made_up(secret)", "unknown"),
])
def test_early_error_result_uses_existing_typed_failure_descriptor(failure, expected):
    result = {"failed": True, "failure_reason": failure, "error": "secret-detail", "api_calls": 3}
    metadata = agent_end_metadata(result)
    assert metadata == {"turn_exit_reason": expected, "api_call_count": 3, "api_call_count_kind": "legacy_iteration", "stale": False}


@pytest.mark.parametrize("count,expected", [(True, 0), (False, 0), (1.5, 0), (float("inf"), 0), (10**30, 1_000_000)])
def test_count_rejects_non_integer_values_and_bounds_the_artifact(count, expected):
    assert agent_end_metadata({"api_calls": count})["api_call_count"] == expected


@pytest.mark.parametrize("exhausted,expected", [(True, "context_compression_exhausted"), (False, "unknown"),
                                               ("true", "unknown"), (None, "unknown")])
def test_only_typed_exhausted_compression_is_upgraded_from_overflow(exhausted, expected):
    metadata = agent_end_metadata({"failure_reason": "context_overflow", "compression_exhausted": exhausted})
    assert metadata["turn_exit_reason"] == expected
