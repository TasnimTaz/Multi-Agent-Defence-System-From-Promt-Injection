"""
ablation_summary.py (v2 - content-based matching)

Instead of guessing AgentDojo's on-disk directory layout (which varies by
version/config), this version recursively scans --runs-dir for ALL *.json
files and filters by the fields actually present INSIDE each result file:
  - "suite_name"      e.g. "workspace"
  - "pipeline_name"   e.g. "openai-compatible-macd_defense" or "openai-compatible"
  - "injection_task_id"  (non-null => this is a security test case)
  - "security"        boolean, True == attacker's goal was met (AgentDojo semantics)

Usage:
    python ablation_summary.py --runs-dir . --suite workspace

If your files aren't found, first run:
    python ablation_summary.py --debug --runs-dir .
to print every JSON file found and its (suite_name, pipeline_name) so you can
confirm the pipeline_name strings match what's expected below.
"""
import argparse
import json
from pathlib import Path

# label -> defense suffix used when building the pipeline_name that AgentDojo
# writes into each result JSON (llm_name is prefixed automatically, e.g.
# "openai-compatible-macd_defense"; baseline has no suffix at all).
DEFENSES = {
    "Baseline (no defense)": None,
    "Tool Guard only": "macd_guard_only",
    "Multi-Agent detector only": "macd_detector_only",
    "Dual-layer (Guard + Detector)": "macd_defense",
}


def pipeline_name_for(defense: str | None) -> str:
    return f"openai-compatible-{defense}" if defense else "openai-compatible"


def load_all_results(runs_dir: Path) -> list[dict]:
    results = []
    for f in runs_dir.rglob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "pipeline_name" in data:
            data["_source_file"] = str(f)
            results.append(data)
    return results


def compute_asr(results: list[dict], suite: str, pipeline_name: str) -> tuple[int, int, float]:
    total = 0
    succeeded = 0
    for data in results:
        if data.get("suite_name") != suite:
            continue
        if data.get("pipeline_name") != pipeline_name:
            continue
        if data.get("injection_task_id") is None:
            continue  # utility-only case, not a security test case
        security = data.get("security")
        if security is None:
            continue
        total += 1
        if security:  # True == attacker's goal WAS met (AgentDojo paper Sec 3.4)
            succeeded += 1
    asr = (succeeded / total * 100) if total else float("nan")
    return succeeded, total, asr


def compute_utility(results: list[dict], suite: str, pipeline_name: str) -> tuple[int, int, float]:
    total = 0
    passed = 0
    for data in results:
        if data.get("suite_name") != suite or data.get("pipeline_name") != pipeline_name:
            continue
        utility = data.get("utility")
        if utility is None:
            continue
        total += 1
        if utility:
            passed += 1
    rate = (passed / total * 100) if total else float("nan")
    return passed, total, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("."))
    parser.add_argument("--suite", type=str, default="workspace")
    parser.add_argument("--out", type=Path, default=Path("ablation_summary.md"))
    parser.add_argument("--debug", action="store_true", help="List every result JSON found and its suite/pipeline, then exit.")
    args = parser.parse_args()

    all_results = load_all_results(args.runs_dir)

    if args.debug:
        print(f"Found {len(all_results)} result JSON file(s) under {args.runs_dir.resolve()}:\n")
        for data in all_results:
            print(f"  suite_name={data.get('suite_name')!r:20} pipeline_name={data.get('pipeline_name')!r:35} "
                  f"user_task_id={data.get('user_task_id')!r:15} injection_task_id={data.get('injection_task_id')!r:20} "
                  f"utility={data.get('utility')!r:6} security={data.get('security')!r:6} "
                  f"file={data['_source_file']}")
        return

    rows = []
    for label, defense in DEFENSES.items():
        pname = pipeline_name_for(defense)
        succ, total, asr = compute_asr(all_results, args.suite, pname)
        u_pass, u_total, u_rate = compute_utility(all_results, args.suite, pname)
        rows.append((label, succ, total, asr, u_pass, u_total, u_rate))

    header = f"{'Configuration':<32} {'Sec.Succ':>9} {'Sec.Tot':>8} {'ASR %':>7}   {'Util.Pass':>10} {'Util.Tot':>9} {'Util %':>7}"
    lines = [header, "-" * len(header)]
    for label, succ, total, asr, u_pass, u_total, u_rate in rows:
        asr_str = f"{asr:.1f}" if total else "N/A"
        u_str = f"{u_rate:.1f}" if u_total else "N/A"
        lines.append(f"{label:<32} {succ:>9} {total:>8} {asr_str:>7}   {u_pass:>10} {u_total:>9} {u_str:>7}")
    table_text = "\n".join(lines)
    print(table_text)

    if not any(total for _, _, total, *_ in rows):
        print("\n⚠️  No matching result files found for ANY configuration.")
        print("    Run with --debug to see what suite_name/pipeline_name values actually exist in your JSON files:")
        print(f"    python {Path(__file__).name} --debug --runs-dir {args.runs_dir}")
        return

    md_lines = [
        "| Configuration | Security Succeeded | Security Total | ASR % | Utility Passed | Utility Total | Utility % |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, succ, total, asr, u_pass, u_total, u_rate in rows:
        asr_str = f"{asr:.1f}" if total else "N/A"
        u_str = f"{u_rate:.1f}" if u_total else "N/A"
        md_lines.append(f"| {label} | {succ} | {total} | {asr_str} | {u_pass} | {u_total} | {u_str} |")
    args.out.write_text("\n".join(md_lines))
    print(f"\nMarkdown table written to {args.out}")


if __name__ == "__main__":
    main()