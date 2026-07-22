import sys
import types
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, MACD_DEFENSE_LAYERS, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.scripts.ablation import (
    _build_report,
    _classify_failure,
    exact_mcnemar_p,
    factorial_effects,
    paired_comparison,
    summarize_outcomes,
    wilson_interval,
)
from agentdojo.types import ChatMessage


class FakeLLM(BasePipelineElement):
    name = "fake-llm"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        return query, runtime, env, messages, extra_args


class FakeGuard(FakeLLM):
    pass


class FakeDetector(FakeLLM):
    def __init__(self, raise_on_injection: bool):
        self.raise_on_injection = raise_on_injection


@pytest.fixture
def fake_macd_modules(monkeypatch):
    guard_module = types.ModuleType("agents.macd_tool_call_guard")
    guard_module.MACDToolCallGuard = FakeGuard
    detector_module = types.ModuleType("agentdojo.agent_pipeline.macd_defense")
    detector_module.MACDPromptInjectionDetector = FakeDetector
    monkeypatch.setitem(sys.modules, "agents.macd_tool_call_guard", guard_module)
    monkeypatch.setitem(sys.modules, "agentdojo.agent_pipeline.macd_defense", detector_module)


@pytest.mark.parametrize(
    ("defense", "expected_types"),
    [
        ("macd_tool_guard", [FakeGuard, ToolsExecutor, FakeLLM]),
        ("macd_multi_agent", [ToolsExecutor, FakeDetector, FakeLLM]),
        ("macd_defense", [FakeGuard, ToolsExecutor, FakeDetector, FakeLLM]),
    ],
)
def test_macd_ablation_pipeline_layers(fake_macd_modules, defense, expected_types):
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=FakeLLM(),
            model_id=None,
            defense=defense,
            system_message_name=None,
            system_message="test system message",
        )
    )

    tools_loop = pipeline.elements[-1]
    assert isinstance(tools_loop, ToolsExecutionLoop)
    assert [type(element) for element in tools_loop.elements] == expected_types
    assert pipeline.name == f"fake-llm-{defense}"


def test_ablation_layer_mapping_is_full_factorial():
    assert MACD_DEFENSE_LAYERS == {
        "macd_tool_guard": (True, False),
        "macd_multi_agent": (False, True),
        "macd_defense": (True, True),
    }


def test_summarize_outcomes_treats_true_as_attack_success():
    summary = summarize_outcomes([True, False, False, False], [True, True, False, True])

    assert summary["total_attack_cases"] == 4
    assert summary["attack_successes"] == 1
    assert summary["attack_success_rate"] == 0.25
    assert summary["attack_blocked_cases"] == 3
    assert summary["attack_block_rate"] == 0.75
    assert summary["utility_rate"] == 0.75
    assert summary["attack_success_rate_wilson_95"] == pytest.approx(wilson_interval(1, 4))


def test_summarize_outcomes_reports_zero_asr_when_all_attacks_fail():
    summary = summarize_outcomes([False, False], [True, True])

    assert summary["attack_successes"] == 0
    assert summary["attack_success_rate"] == 0.0
    assert summary["attack_block_rate"] == 1.0


def test_factorial_effects_use_attack_success_rate():
    summaries = {
        "baseline": {"attack_success_rate": 0.8},
        "tool_guard_only": {"attack_success_rate": 0.5},
        "multi_agent_only": {"attack_success_rate": 0.6},
        "dual_layer": {"attack_success_rate": 0.2},
    }

    effects = factorial_effects(summaries)

    assert effects["tool_guard_reduction_without_multi_agent_pp"] == pytest.approx(30.0)
    assert effects["tool_guard_reduction_with_multi_agent_pp"] == pytest.approx(40.0)
    assert effects["multi_agent_reduction_without_tool_guard_pp"] == pytest.approx(20.0)
    assert effects["multi_agent_reduction_with_tool_guard_pp"] == pytest.approx(30.0)
    assert effects["interaction_difference_in_differences_pp"] == pytest.approx(-10.0)


def test_paired_comparison_counts_direction_and_exact_p_value():
    comparison = {(1,): False, (2,): False, (3,): True, (4,): True}
    dual = {(1,): True, (2,): True, (3,): True, (4,): False}

    result = paired_comparison(comparison, dual)

    assert result["paired_cases"] == 4
    assert result["dual_layer_improvements"] == 1
    assert result["dual_layer_regressions"] == 2
    assert result["exact_mcnemar_p"] == exact_mcnemar_p(1, 2)


def test_classify_failure_maps_provider_limit_errors():
    assert _classify_failure(RuntimeError("Error code: 413 Request too large")) == "provider_request_too_large"
    assert _classify_failure(RuntimeError("Error code: 429 rate_limit_exceeded")) == "provider_rate_limit"
    assert _classify_failure(RuntimeError("Missing credentials. Set OPENAI_API_KEY")) == "missing_credentials"


def test_build_report_marks_partial_failures():
    report = _build_report(
        created_utc=datetime(2026, 7, 18, tzinfo=timezone.utc),
        model="openai-compatible",
        model_id="llama-3.1-8b-instant",
        benchmark_version="v1.2.2",
        suites=("workspace",),
        user_tasks=("user_task_0",),
        injection_tasks=("injection_task_0",),
        attack="important_instructions",
        repetitions=1,
        order_seed=42,
        reuse_results=False,
        selected=["baseline", "tool_guard_only"],
        execution_order=[],
        attack_outcomes_by_configuration={"baseline": {(1, "workspace", "u", "i"): False}, "tool_guard_only": {}},
        utility_by_configuration={"baseline": {(1, "workspace", "u", "i"): True}, "tool_guard_only": {}},
        elapsed_by_configuration={"baseline": 1.0, "tool_guard_only": 0.5},
        per_run=[{"configuration": "baseline"}],
        failures=[
            {
                "repetition": 1,
                "suite": "workspace",
                "configuration": "tool_guard_only",
                "kind": "provider_request_too_large",
                "error_type": "APIStatusError",
                "message": "Error code: 413",
            }
        ],
    )

    assert report["metadata"]["status"] == "partial_failure"
    assert report["metadata"]["failure_count"] == 1
    assert report["aggregate"]["baseline"]["attack_success_rate"] == 0.0
    assert report["aggregate"]["tool_guard_only"]["attack_success_rate"] is None
