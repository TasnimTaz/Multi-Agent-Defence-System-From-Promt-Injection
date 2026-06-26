# agents/guard.py

import json
from config import MODEL_NAME, GUARD_SYSTEM_PROMPT, SAFE_REFUSAL_MSG, GROQ_CLIENT


class GuardAgent:
    """
    Post-output validation agent using Groq.
    Paper: 'Guard Agent' in Fig. 1 and Fig. 3.
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = MODEL_NAME
        self.system_prompt = GUARD_SYSTEM_PROMPT
        print("[GuardAgent] Ready via Groq")

    def get_refusal(self) -> str:
        return SAFE_REFUSAL_MSG

    def validate(self, response_text: str) -> tuple[bool, str, str]:
        """
        Returns: (is_safe, cleaned_response, reason)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": f"Validate this AI response:\n\n{response_text}"},
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                stream=False
            )
            raw = completion.choices[0].message.content.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            is_safe  = bool(result.get("is_safe", False))
            reason   = result.get("reason", "")
            cleaned  = result.get("cleaned_response", "")

            return is_safe, cleaned, reason

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fail-safe: if guard fails, block the output
            return False, "", f"Guard error/fail-safe: {str(e)}"