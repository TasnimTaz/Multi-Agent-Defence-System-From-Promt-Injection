# agents/domain_llm.py

from config import TARGET_MODEL, SYSTEM_PROMPT, GROQ_CLIENT
from agents.groq_utils import safe_completion


class DomainLLM:
    """
    Main LLM agent that processes legitimate user queries using Groq.
    Paper: 'Domain LLM Agent' in Fig. 1 and Fig. 2.
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = TARGET_MODEL
        self.system_prompt = SYSTEM_PROMPT
        print(f"[DomainLLM] Ready via Groq — model: {self.model}")

    def generate(self, user_input: str) -> str:
        try:
            completion = safe_completion(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_input},
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"DomainLLM Error: {str(e)}"