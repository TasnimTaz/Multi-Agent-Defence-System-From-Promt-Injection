# agents/domain_llm.py

from config import MODEL_NAME, SYSTEM_PROMPT, GROQ_CLIENT


class DomainLLM:
    """
    Main LLM agent that processes legitimate user queries using Groq.
    Paper: 'Domain LLM Agent' in Fig. 1 and Fig. 2.
    """

    def __init__(self):
        self.client = GROQ_CLIENT
        self.model = MODEL_NAME
        self.system_prompt = SYSTEM_PROMPT
        print(f"[DomainLLM] Ready via Groq — model: {self.model}")

    def generate(self, user_input: str) -> str:
        try:
            completion = self.client.chat.completions.create(
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