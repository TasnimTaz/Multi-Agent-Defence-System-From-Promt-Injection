# agents/coordinator.py

import json
# config থেকে DEFENSE_MODEL ইম্পোর্ট করা হয়েছে
from config import DEFENSE_MODEL, COORDINATOR_SYSTEM_PROMPT, GROQ_CLIENT


class CoordinatorAgent:
    """
    Pre-input screening agent using Groq.
    Paper: 'Coordinator' in Fig. 2 and Fig. 3.
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = DEFENSE_MODEL
        self.system_prompt = COORDINATOR_SYSTEM_PROMPT
        print(f"[CoordinatorAgent] Ready via Groq using {self.model}")

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
                temperature=0.1,  
                max_completion_tokens=4096, 
                stream=False
            )
            raw = completion.choices[0].message.content.strip()

            # 🛠️ ট্রিক ১: রিজনিং মডেলের <think>...</think> ট্যাগ থাকলে তা বাদ দেওয়া
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            # 🛠️ ট্রিক ২: ব্যাকটিক বা ```json ফরম্যাট থাকলে তা পরিষ্কার করা
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            # ফাইনাল জেসন পার্সিং
            result = json.loads(raw)
            is_safe   = bool(result.get("is_safe", False))
            category  = result.get("category", "unknown")
            reason    = result.get("reason", "")
            confidence = float(result.get("confidence", 1.0))

            return is_safe, category, reason, confidence

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fail-safe: if classifier fails, block the input
            return False, "unknown", f"Coordinator error/fail-safe: {str(e)}", 1.0