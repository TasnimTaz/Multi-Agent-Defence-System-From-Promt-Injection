# agents/macd_pattern_expert.py

import json
from config import DEFENSE_MODEL, MACD_PATTERN_SYSTEM_PROMPT, GROQ_CLIENT
from agents.groq_utils import safe_completion


class PatternExpertAgent:
    """
    MACD Agent 1 (re-mapped from MACD paper's Technical Defense Expert).
    Paper: Li et al., "MACD: Multi-Agent Collaborative Approach for
    Cybersecurity Defense Strategy Generation" — Sec. 4.4, Agent 1.

    এখানে "technique-specific mitigation" এর বদলে "syntax/encoding-level
    attack signal" ডিটেক্ট করা হয় — base64/hex/zero-width/delimiter-breaking।
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = DEFENSE_MODEL
        self.system_prompt = MACD_PATTERN_SYSTEM_PROMPT
        print(f"[PatternExpertAgent] Ready via Groq using {self.model}")

    @staticmethod
    def _clean_json(raw: str) -> str:
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def analyze(self, user_input: str) -> dict:
        """
        Returns: {"is_suspicious": bool, "patterns_found": [...], "confidence": float, "reason": str}
        """
        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this input for syntactic/technical attack patterns:\n\n{user_input}"},
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False,
            )
            raw = self._clean_json(completion.choices[0].message.content.strip())
            result = json.loads(raw)

            return {
                "agent": "pattern_expert",
                "is_suspicious": bool(result.get("is_suspicious", False)),
                "patterns_found": result.get("patterns_found", []),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
            }
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fail-safe: agent error হলে suspicious ধরে নেওয়া হয়, Judge চূড়ান্ত সিদ্ধান্ত নেবে
            return {
                "agent": "pattern_expert",
                "is_suspicious": True,
                "patterns_found": [],
                "confidence": 1.0,
                "reason": f"PatternExpertAgent error/fail-safe: {str(e)}",
            }