# MACD component ablation

The ablation runner evaluates the same AgentDojo cases under a 2x2 design:

| Configuration | Pre-execution Tool Guard | Post-output Multi-Agent detector |
|---|:---:|:---:|
| Baseline | no | no |
| Tool Guard only | yes | no |
| Multi-Agent only | no | yes |
| Dual-layer | yes | yes |

The existing `macd_defense` behavior remains the dual-layer system. The two new defense names, `macd_tool_guard`
and `macd_multi_agent`, expose one layer at a time without modifying either layer's implementation.

## Deadline-friendly pilot

Use an identical, pre-declared subset for all four configurations. This example runs four task pairs and stores
everything outside the existing `runs` directory:

```powershell
$env:PYTHONPATH = "src;."
.venv\Scripts\python.exe -m agentdojo.scripts.ablation `
  --suite workspace `
  --user-task user_task_0 --user-task user_task_1 `
  --injection-task injection_task_0 --injection-task injection_task_1 `
  --attack important_instructions `
  --model gpt-4o-2024-05-13 `
  --run-name paper_pilot
```

The `PYTHONPATH` line makes the command use this checkout's source tree; it can be omitted after an editable project
install.

Omit the task filters for the complete workspace suite. Add `--repetitions 3` or more when budget permits because
LLM outputs are stochastic. Configurations are executed in a reproducibly shuffled order (`--order-seed 42`) to
reduce order bias. To continue an interrupted named run without paying again for completed cases, add
`--reuse-results`.

The runner writes the raw AgentDojo traces plus three paper-ready reports:

- `ablation_results.json`: metadata, execution order, per-run results, aggregate metrics, factorial effects, and
  paired comparisons.
- `ablation_results.csv`: one aggregate row per configuration.
- `ablation_results.md`: a table suitable for adaptation into the paper.

## What to report

Report attack success rate (ASR) together with its numerator, denominator, and 95% Wilson interval. Also report
AgentDojo utility: an apparent security gain is not meaningful if the defense blocks legitimate task completion.
The report includes paired exact McNemar tests against the dual layer because every configuration sees the same
task pairs.

AgentDojo's raw result field is unfortunately named `security`: for injection tasks, `security=true` means the
injected attacker's goal was achieved. The ablation runner therefore counts raw `true` values as attack successes.
It reports **Attack Block Rate = 1 - ASR** instead of displaying the ambiguous label "Security." Never interpret
`security=true` as a successful defense.

For the factorial effects, positive "reduction" values mean fewer successful attacks. The interaction
difference-in-differences is negative when the dual layer reduces ASR beyond the sum of the two isolated
reductions. With a small pilot, describe all effect sizes as preliminary and avoid treating a non-significant test
as proof that two configurations are equivalent.

Keep the target model, attack, AgentDojo version, task subset, system prompt, tool-output format, defense-agent
models, temperatures, and retry policy fixed across configurations. Archive the JSON metadata and task traces with
the paper artifact.
