# agents/coordinator.py

import json
from config import MODEL_NAME, COORDINATOR_SYSTEM_PROMPT, GROQ_CLIENT


class CoordinatorAgent:
    """
    Pre-input screening agent using Groq.
    Paper: 'Coordinator' in Fig. 2 and Fig. 3.
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = MODEL_NAME
        self.system_prompt = COORDINATOR_SYSTEM_PROMPT
        print("[CoordinatorAgent] Ready via Groq")

    def classify(self, user_input: str) -> tuple[bool, str, str, float]:
        """
        Returns: (is_safe, category, reason, confidence)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": f"Classify this input:\n\n{user_input}"},
                ],
                temperature=0.1,  # জেসন ফরম্যাট ফিক্সড রাখার জন্য কম টেম্পারেচার
                max_completion_tokens=512,
                stream=False
            )
            raw = completion.choices[0].message.content.strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            is_safe   = bool(result.get("is_safe", False))
            category  = result.get("category", "unknown")
            reason    = result.get("reason", "")
            confidence = float(result.get("confidence", 1.0))

            return is_safe, category, reason, confidence

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fail-safe: if classifier fails, block the input
            return False, "unknown", f"Coordinator error/fail-safe: {str(e)}", 1.0