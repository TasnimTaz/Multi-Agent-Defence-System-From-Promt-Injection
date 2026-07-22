"""Run and report the four-configuration MACD component ablation.

This module deliberately uses AgentDojo's existing benchmark functions rather
than introducing a second evaluator. Every configuration therefore receives
the same tasks, attacks, utility checks, and security semantics.
"""

import csv
import json
import math
import platform
import random
import sys
import time
import traceback
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click

from agentdojo.benchmark import SuiteResults
from agentdojo.models import ModelsEnum
from agentdojo.scripts.benchmark import benchmark_suite
from agentdojo.task_suite.load_suites import get_suite

ABLATION_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Baseline",
        "defense": None,
        "tool_guard": False,
        "multi_agent": False,
    },
    "tool_guard_only": {
        "label": "Tool Guard only",
        "defense": "macd_tool_guard",
        "tool_guard": True,
        "multi_agent": False,
    },
    "multi_agent_only": {
        "label": "Multi-Agent only",
        "defense": "macd_multi_agent",
        "tool_guard": False,
        "multi_agent": True,
    },
    "dual_layer": {
        "label": "Dual-layer",
        "defense": "macd_defense",
        "tool_guard": True,
        "multi_agent": True,
    },
}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


def summarize_outcomes(attack_goal_achieved: Iterable[bool], utility: Iterable[bool]) -> dict[str, Any]:
    """Summarize outcomes using AgentDojo's injection-task boolean semantics.

    AgentDojo stores an injection task's result in a field named ``security``,
    but ``True`` means that the injected attacker's goal was achieved.  It is
    therefore an attack success, not a secure outcome.
    """
    attack_outcomes = list(attack_goal_achieved)
    utility_values = list(utility)
    total = len(attack_outcomes)
    attack_successes = sum(attack_outcomes)
    attack_blocked = total - attack_successes
    utility_total = len(utility_values)
    utility_successes = sum(utility_values)
    return {
        "total_attack_cases": total,
        "attack_successes": attack_successes,
        "attack_success_rate": attack_successes / total if total else None,
        "attack_success_rate_wilson_95": wilson_interval(attack_successes, total),
        "attack_blocked_cases": attack_blocked,
        "attack_block_rate": attack_blocked / total if total else None,
        "utility_total": utility_total,
        "utility_successes": utility_successes,
        "utility_rate": utility_successes / utility_total if utility_total else None,
    }


def exact_mcnemar_p(improvements: int, regressions: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(improvements, regressions) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_comparison(
    comparison: Mapping[tuple[Any, ...], bool], reference: Mapping[tuple[Any, ...], bool]
) -> dict[str, Any]:
    """Compare attack-success booleans with the dual layer on identical cases."""
    common_keys = comparison.keys() & reference.keys()
    improvements = sum(comparison[key] and not reference[key] for key in common_keys)
    regressions = sum(not comparison[key] and reference[key] for key in common_keys)
    return {
        "paired_cases": len(common_keys),
        "dual_layer_improvements": improvements,
        "dual_layer_regressions": regressions,
        "exact_mcnemar_p": exact_mcnemar_p(improvements, regressions),
    }


def factorial_effects(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Compute transparent percentage-point effects for the 2x2 design."""
    required = set(ABLATION_CONFIGURATIONS)
    if not required.issubset(summaries):
        return {}
    rates = {name: summaries[name]["attack_success_rate"] for name in required}
    if any(rate is None for rate in rates.values()):
        return {}

    baseline = rates["baseline"]
    guard = rates["tool_guard_only"]
    multi = rates["multi_agent_only"]
    dual = rates["dual_layer"]
    return {
        "tool_guard_reduction_without_multi_agent_pp": 100 * (baseline - guard),
        "tool_guard_reduction_with_multi_agent_pp": 100 * (multi - dual),
        "multi_agent_reduction_without_tool_guard_pp": 100 * (baseline - multi),
        "multi_agent_reduction_with_tool_guard_pp": 100 * (guard - dual),
        "tool_guard_average_main_effect_pp": 50 * ((baseline - guard) + (multi - dual)),
        "multi_agent_average_main_effect_pp": 50 * ((baseline - multi) + (guard - dual)),
        "interaction_difference_in_differences_pp": 100 * (dual - guard - multi + baseline),
    }


def _flatten_results(
    destination: dict[tuple[Any, ...], bool], repetition: int, suite_name: str, values: Mapping[tuple[str, str], bool]
) -> None:
    for (user_task, injection_task), outcome in values.items():
        destination[(repetition, suite_name, user_task, injection_task)] = outcome


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _classify_failure(error: BaseException) -> str:
    message = str(error)
    if "Error code: 413" in message or "Request too large" in message:
        return "provider_request_too_large"
    if "Error code: 429" in message or "rate_limit_exceeded" in message or "Too Many Requests" in message:
        return "provider_rate_limit"
    if "Missing credentials" in message or "OPENAI_API_KEY" in message or "GROQ_API_KEY" in message:
        return "missing_credentials"
    return type(error).__name__


def _build_report(
    *,
    created_utc: datetime,
    model: str,
    model_id: str | None,
    benchmark_version: str,
    suites: tuple[str, ...],
    user_tasks: tuple[str, ...],
    injection_tasks: tuple[str, ...],
    attack: str,
    repetitions: int,
    order_seed: int,
    reuse_results: bool,
    selected: list[str],
    execution_order: list[dict[str, Any]],
    attack_outcomes_by_configuration: Mapping[str, Mapping[tuple[Any, ...], bool]],
    utility_by_configuration: Mapping[str, Mapping[tuple[Any, ...], bool]],
    elapsed_by_configuration: Mapping[str, float],
    per_run: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {}
    for name in selected:
        summary = summarize_outcomes(
            attack_outcomes_by_configuration[name].values(), utility_by_configuration[name].values()
        )
        aggregate[name] = {**summary, "elapsed_seconds": elapsed_by_configuration[name]}

    paired = {}
    if "dual_layer" in selected:
        for name in selected:
            if name != "dual_layer":
                paired[name] = paired_comparison(
                    attack_outcomes_by_configuration[name], attack_outcomes_by_configuration["dual_layer"]
                )

    return {
        "metadata": {
            "created_utc": created_utc.isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "model": model,
            "model_id": model_id,
            "benchmark_version": benchmark_version,
            "suites": list(suites),
            "user_tasks": list(user_tasks),
            "injection_tasks": list(injection_tasks),
            "attack": attack,
            "repetitions": repetitions,
            "order_seed": order_seed,
            "reuse_results": reuse_results,
            "agentdojo_security_field_semantics": "true means the injected attacker goal was achieved",
            "status": "partial_failure" if failures else "complete",
            "completed_run_count": len(per_run),
            "failure_count": len(failures),
        },
        "execution_order": execution_order,
        "configuration_definitions": {name: ABLATION_CONFIGURATIONS[name] for name in selected},
        "aggregate": aggregate,
        "per_run": per_run,
        "factorial_effects": factorial_effects(aggregate),
        "paired_comparisons": paired,
        "failures": failures,
    }


def _write_reports(report: dict[str, Any], run_dir: Path) -> tuple[Path, Path, Path]:
    json_path = run_dir / "ablation_results.json"
    csv_path = run_dir / "ablation_results.csv"
    markdown_path = run_dir / "ablation_results.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "configuration",
            "label",
            "tool_guard",
            "multi_agent",
            "total_attack_cases",
            "attack_successes",
            "attack_success_rate",
            "asr_wilson_95_low",
            "asr_wilson_95_high",
            "attack_blocked_cases",
            "attack_block_rate",
            "utility_rate",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for name, summary in report["aggregate"].items():
            interval = summary["attack_success_rate_wilson_95"] or (None, None)
            configuration = ABLATION_CONFIGURATIONS[name]
            writer.writerow(
                {
                    "configuration": name,
                    "label": configuration["label"],
                    "tool_guard": configuration["tool_guard"],
                    "multi_agent": configuration["multi_agent"],
                    "total_attack_cases": summary["total_attack_cases"],
                    "attack_successes": summary["attack_successes"],
                    "attack_success_rate": summary["attack_success_rate"],
                    "asr_wilson_95_low": interval[0],
                    "asr_wilson_95_high": interval[1],
                    "attack_blocked_cases": summary["attack_blocked_cases"],
                    "attack_block_rate": summary["attack_block_rate"],
                    "utility_rate": summary["utility_rate"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )

    lines = [
        "# MACD component ablation",
        "",
        f"Generated: {report['metadata']['created_utc']}",
        "",
        f"Run status: `{report['metadata']['status']}`",
        "",
        "AgentDojo raw `security=true` means the injected attacker goal was achieved. ASR below follows that "
        "benchmark convention; Attack Block Rate is `1 - ASR`.",
        "",
    ]
    if report.get("failures"):
        lines.extend(["## Partial run notice", ""])
        for failure in report["failures"]:
            lines.append(
                f"- Repetition {failure['repetition']}, suite `{failure['suite']}`, configuration "
                f"`{ABLATION_CONFIGURATIONS[failure['configuration']]['label']}` failed with "
                f"`{failure['kind']}`: {failure['message']}"
            )
        lines.append("")
    lines.extend(
        [
            "| Configuration | Tool Guard | Multi-Agent | N | Attack successes | ASR (95% Wilson CI) | Attack Block Rate | Utility |",
            "|---|:---:|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in report["aggregate"].items():
        configuration = ABLATION_CONFIGURATIONS[name]
        interval = summary["attack_success_rate_wilson_95"]
        interval_text = "n/a" if interval is None else f"[{100 * interval[0]:.2f}%, {100 * interval[1]:.2f}%]"
        lines.append(
            f"| {configuration['label']} | {'yes' if configuration['tool_guard'] else 'no'} "
            f"| {'yes' if configuration['multi_agent'] else 'no'} | {summary['total_attack_cases']} "
            f"| {summary['attack_successes']} | {_rate(summary['attack_success_rate'])} {interval_text} "
            f"| {_rate(summary['attack_block_rate'])} | {_rate(summary['utility_rate'])} |"
        )

    if report["factorial_effects"]:
        lines.extend(["", "## Factorial effects", ""])
        for name, value in report["factorial_effects"].items():
            lines.append(f"- {name}: {value:.2f} percentage points")
        lines.extend(
            [
                "",
                "Negative interaction difference-in-differences indicates that the combined system reduces ASR more "
                "than the sum of the two single-layer reductions; interpret this only with its paired outcomes and "
                "sample size.",
            ]
        )

    if report["paired_comparisons"]:
        lines.extend(["", "## Paired comparisons against dual-layer", ""])
        for name, comparison in report["paired_comparisons"].items():
            lines.append(
                f"- {ABLATION_CONFIGURATIONS[name]['label']}: "
                f"improvements={comparison['dual_layer_improvements']}, "
                f"regressions={comparison['dual_layer_regressions']}, "
                f"exact McNemar p={comparison['exact_mcnemar_p']:.6f}"
            )

    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, markdown_path


@click.command()
@click.option("--suite", "suites", multiple=True, default=("workspace",), show_default=True)
@click.option("--user-task", "user_tasks", "-ut", multiple=True)
@click.option("--injection-task", "injection_tasks", "-it", multiple=True)
@click.option("--attack", default="important_instructions", show_default=True)
@click.option("--model", type=click.Choice([value.value for value in ModelsEnum]), default="gpt-4o-2024-05-13")
@click.option("--model-id", default=None)
@click.option("--benchmark-version", default="v1.2.2", show_default=True)
@click.option("--tool-delimiter", default="tool", show_default=True)
@click.option("--tool-output-format", type=click.Choice(["yaml", "json"]), default=None)
@click.option("--system-message-name", default=None)
@click.option("--system-message", default=None)
@click.option(
    "--configuration",
    "configurations",
    multiple=True,
    type=click.Choice(list(ABLATION_CONFIGURATIONS)),
    help="Run a subset; by default all four configurations are evaluated.",
)
@click.option("--repetitions", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--order-seed", type=int, default=42, show_default=True)
@click.option("--output-dir", type=Path, default=Path("./ablation_runs"), show_default=True)
@click.option("--run-name", default=None, help="Stable directory name, useful for resuming an interrupted run.")
@click.option("--reuse-results", is_flag=True, help="Reuse matching task logs instead of rerunning them.")
def main(
    suites: tuple[str, ...],
    user_tasks: tuple[str, ...],
    injection_tasks: tuple[str, ...],
    attack: str,
    model: str,
    model_id: str | None,
    benchmark_version: str,
    tool_delimiter: str,
    tool_output_format: str | None,
    system_message_name: str | None,
    system_message: str | None,
    configurations: tuple[str, ...],
    repetitions: int,
    order_seed: int,
    output_dir: Path,
    run_name: str | None,
    reuse_results: bool,
) -> None:
    """Run baseline, each MACD layer alone, and the complete dual layer."""
    selected = list(dict.fromkeys(configurations or ABLATION_CONFIGURATIONS.keys()))
    created_utc = datetime.now(timezone.utc)
    run_name = run_name or created_utc.strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    attack_outcomes_by_configuration: dict[str, dict[tuple[Any, ...], bool]] = {name: {} for name in selected}
    utility_by_configuration: dict[str, dict[tuple[Any, ...], bool]] = {name: {} for name in selected}
    elapsed_by_configuration = {name: 0.0 for name in selected}
    per_run: list[dict[str, Any]] = []
    execution_order: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        order = selected.copy()
        random.Random(order_seed + repetition).shuffle(order)
        execution_order.append({"repetition": repetition + 1, "configurations": order})
        repetition_logdir = run_dir / "task_logs" / f"repetition_{repetition + 1}"

        for configuration_name in order:
            configuration = ABLATION_CONFIGURATIONS[configuration_name]
            for suite_name in suites:
                suite = get_suite(benchmark_version, suite_name)
                started = time.perf_counter()
                try:
                    results: SuiteResults = benchmark_suite(
                        suite=suite,
                        model=ModelsEnum(model),
                        logdir=repetition_logdir,
                        force_rerun=not reuse_results,
                        benchmark_version=benchmark_version,
                        user_tasks=user_tasks,
                        injection_tasks=injection_tasks,
                        model_id=model_id,
                        attack=attack,
                        defense=configuration["defense"],
                        tool_delimiter=tool_delimiter,
                        system_message_name=system_message_name,
                        system_message=system_message,
                        tool_output_format=tool_output_format,  # type: ignore[arg-type]
                    )
                except Exception as error:
                    elapsed_by_configuration[configuration_name] += time.perf_counter() - started
                    failures.append(
                        {
                            "repetition": repetition + 1,
                            "suite": suite_name,
                            "configuration": configuration_name,
                            "kind": _classify_failure(error),
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    report = _build_report(
                        created_utc=created_utc,
                        model=model,
                        model_id=model_id,
                        benchmark_version=benchmark_version,
                        suites=suites,
                        user_tasks=user_tasks,
                        injection_tasks=injection_tasks,
                        attack=attack,
                        repetitions=repetitions,
                        order_seed=order_seed,
                        reuse_results=reuse_results,
                        selected=selected,
                        execution_order=execution_order,
                        attack_outcomes_by_configuration=attack_outcomes_by_configuration,
                        utility_by_configuration=utility_by_configuration,
                        elapsed_by_configuration=elapsed_by_configuration,
                        per_run=per_run,
                        failures=failures,
                    )
                    paths = _write_reports(report, run_dir)
                    click.echo("Ablation interrupted. Partial results saved:")
                    for path in paths:
                        click.echo(f"  {path}")
                    click.echo(
                        "Failure summary: "
                        f"configuration={ABLATION_CONFIGURATIONS[configuration_name]['label']}, "
                        f"suite={suite_name}, kind={failures[-1]['kind']}"
                    )
                    raise SystemExit(1) from error
                elapsed = time.perf_counter() - started
                elapsed_by_configuration[configuration_name] += elapsed
                summary = summarize_outcomes(results["security_results"].values(), results["utility_results"].values())
                per_run.append(
                    {
                        "repetition": repetition + 1,
                        "suite": suite_name,
                        "configuration": configuration_name,
                        "elapsed_seconds": elapsed,
                        **summary,
                    }
                )
                _flatten_results(
                    attack_outcomes_by_configuration[configuration_name],
                    repetition,
                    suite_name,
                    results["security_results"],
                )
                _flatten_results(
                    utility_by_configuration[configuration_name],
                    repetition,
                    suite_name,
                    results["utility_results"],
                )

    report = _build_report(
        created_utc=created_utc,
        model=model,
        model_id=model_id,
        benchmark_version=benchmark_version,
        suites=suites,
        user_tasks=user_tasks,
        injection_tasks=injection_tasks,
        attack=attack,
        repetitions=repetitions,
        order_seed=order_seed,
        reuse_results=reuse_results,
        selected=selected,
        execution_order=execution_order,
        attack_outcomes_by_configuration=attack_outcomes_by_configuration,
        utility_by_configuration=utility_by_configuration,
        elapsed_by_configuration=elapsed_by_configuration,
        per_run=per_run,
        failures=failures,
    )
    paths = _write_reports(report, run_dir)
    click.echo("Ablation complete. Reports:")
    for path in paths:
        click.echo(f"  {path}")


if __name__ == "__main__":
    main()
