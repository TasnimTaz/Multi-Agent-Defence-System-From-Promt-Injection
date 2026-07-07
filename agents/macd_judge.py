# agents/macd_judge.py

import json
from config import DEFENSE_MODEL, MACD_JUDGE_SYSTEM_PROMPT, GROQ_CLIENT
from agents.groq_utils import safe_completion


class MACDJudgeAgent:
    """
    MACD Agent 4 (re-mapped from MACD paper's Coordinator Agent).
    Paper: Li et al., Sec. 4.4, Agent 4 — dedup, conflict resolution,
    contextualization verification, gap analysis.

    IPI defense-এ এই ৪ টা কাজ হয়ে দাঁড়ায়:
      (1) তিন এজেন্টের verdict merge করা
      (2) disagreement resolve করা (সবচেয়ে specific/high-confidence signal অনুযায়ী)
      (3) false-positive check (legitimate কিন্তু suspicious-দেখতে input যেন block না হয়)
      (4) চূড়ান্ত is_safe + category + confidence + reason তৈরি করা
    """

    def __init__(self, model: str = None):
        self.client = GROQ_CLIENT
        self.model = model or DEFENSE_MODEL
        self.system_prompt = MACD_JUDGE_SYSTEM_PROMPT
        print(f"[MACDJudgeAgent] Ready via Groq using {self.model}")

    @staticmethod
    def _clean_json(raw: str) -> str:
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def synthesize(self, user_input: str, pattern_verdict: dict, intent_verdict: dict, category_verdict: dict) -> tuple[bool, str, str, float]:
        """
        Returns: (is_safe, category, reason, confidence) — same shape as CoordinatorAgent.classify
        so MACDPipeline can plug into the same DomainLLM/GuardAgent flow.
        """
        verdicts_summary = (
            f"Agent 1 (Pattern/Syntax): is_suspicious={pattern_verdict.get('is_suspicious')}, "
            f"patterns={pattern_verdict.get('patterns_found')}, "
            f"confidence={pattern_verdict.get('confidence')}, reason={pattern_verdict.get('reason')}\n"
            f"Agent 2 (Intent/Action): is_malicious_intent={intent_verdict.get('is_malicious_intent')}, "
            f"intent_category={intent_verdict.get('intent_category')}, "
            f"confidence={intent_verdict.get('confidence')}, reason={intent_verdict.get('reason')}\n"
            f"Agent 3 (Attack-Category): matches_known_attack={category_verdict.get('matches_known_attack')}, "
            f"category={category_verdict.get('category')}, "
            f"confidence={category_verdict.get('confidence')}, reason={category_verdict.get('reason')}"
        )
        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Original user input:\n{user_input}\n\n"
                            f"Expert verdicts:\n{verdicts_summary}"
                        ),
                    },
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False,
            )
            raw = self._clean_json(completion.choices[0].message.content.strip())
            result = json.loads(raw)

            is_safe = bool(result.get("is_safe", False))
            category = result.get("category", "unknown")
            reason = result.get("reason", "")
            confidence = float(result.get("confidence", 1.0))

            return is_safe, category, reason, confidence

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fail-safe: Judge fail করলে, কোনো এজেন্ট suspicious flag দিয়েছে কিনা দেখে block করা
            any_flagged = (
                pattern_verdict.get("is_suspicious")
                or intent_verdict.get("is_malicious_intent")
                or category_verdict.get("matches_known_attack")
            )
            return (
                not any_flagged,
                category_verdict.get("category", "unknown") if any_flagged else "safe",
                f"MACDJudgeAgent error/fail-safe (majority of agents): {str(e)}",
                1.0,
            )