# pipelines/chain_pipeline.py

from agents.domain_llm import DomainLLM
from agents.guard import GuardAgent


class ChainPipeline:
    """
    Paper Fig. 1 — Chain-of-Agents Pipeline:

    User Input
        → Domain LLM Agent  (generates candidate response)
        → Guard Agent       (validates output)
        → System Output
    """

    def __init__(self, domain_llm: DomainLLM):
        self.llm   = domain_llm
        self.guard = GuardAgent()

    def run(self, user_input: str) -> dict:
        # Step 1: Domain LLM generates candidate response
        raw_response = self.llm.generate(user_input)

        # Step 2: Guard Agent validates
        is_safe, cleaned, reason = self.guard.validate(raw_response)

        if not is_safe:
            return {
                "pipeline":      "chain",
                "input":         user_input,
                "output":        self.guard.get_refusal(),
                "blocked":       True,
                "block_stage":   "guard",
                "block_reason":  reason,
                "raw_response":  raw_response,
            }

        return {
            "pipeline":     "chain",
            "input":        user_input,
            "output":       cleaned,
            "blocked":      False,
            "block_stage":  None,
            "block_reason": None,
            "raw_response": raw_response,
        }