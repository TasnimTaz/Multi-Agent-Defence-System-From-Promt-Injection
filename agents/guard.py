# agents/guard.py

import json
# config থেকে DEFENSE_MODEL ইম্পোর্ট করা হয়েছে
from config import DEFENSE_MODEL, GUARD_SYSTEM_PROMPT, SAFE_REFUSAL_MSG, GROQ_CLIENT


class GuardAgent:
    """
    Post-output validation agent using Groq.
    Paper: 'Guard Agent' in Fig. 1 and Fig. 3.
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = DEFENSE_MODEL
        self.system_prompt = GUARD_SYSTEM_PROMPT
        print(f"[GuardAgent] Ready via Groq using {self.model}")

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
                max_completion_tokens=4096,  # 🛠️ টোকেন বাড়িয়ে দেওয়া হলো রিজনিং এর জন্য
                stream=False
            )
            raw = completion.choices[0].message.content.strip()

            # 🛠️ ট্রিক ১: রিজনিং মডেলের <think>...</think> ট্যাগ থাকলে তা বাদ দেওয়া
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

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
            # 🛠️ সিনট্যাক্স এররটি ফিক্স করা হলো (কমা ও ব্র্যাকেট ক্লোজিং)
            return False, "", f"Guard error/fail-safe: {str(e)}"