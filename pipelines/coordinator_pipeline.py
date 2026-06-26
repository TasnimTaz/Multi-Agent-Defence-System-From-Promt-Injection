# pipelines/coordinator_pipeline.py

from agents.coordinator import CoordinatorAgent
from agents.domain_llm import DomainLLM
from agents.guard import GuardAgent


class CoordinatorPipeline:
    """
    Paper Fig. 2 & Fig. 3 — Coordinator-based Pipeline:

    User Input
        → API Gateway
        → Event Orchestrator
        → Coordinator
            → [Attack?] YES → Safe Refusal + Logger
            → [Attack?] NO  → Domain LLM
                             → Guard Agent
                             → Buffer-1
                             → Buffer-2
                             → System Output + Logger
    """

    def __init__(self, domain_llm: DomainLLM):
        self.coordinator = CoordinatorAgent()
        self.llm         = domain_llm
        self.guard       = GuardAgent()

    def _api_gateway(self, user_input: str) -> str:
        """Paper Fig. 3: API Gateway — pass-through, logs entry."""
        return user_input

    def _event_orchestrator(self, user_input: str) -> str:
        """Paper Fig. 3: Event Orchestrator — routes to Coordinator."""
        return user_input

    def _buffer(self, response: str, stage: int) -> str:
        """Paper Fig. 3: Buffer-1 and Buffer-2 — additional safety checks."""
        # Enforce 3-bullet rule at buffer stage
        lines = response.split('\n')
        bullet_count = 0
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('-', '*', '•', '·', '1.', '2.', '3.')):
                bullet_count += 1
                if bullet_count > 3:
                    continue
            result.append(line)
        buffered = '\n'.join(result)
        return buffered

    def run(self, user_input: str) -> dict:
        # API Gateway
        user_input = self._api_gateway(user_input)

        # Event Orchestrator
        user_input = self._event_orchestrator(user_input)

        # Coordinator — pre-input classification
        is_safe, category, reason, confidence = self.coordinator.classify(user_input)

        if not is_safe:
            return {
                "pipeline":      "coordinator",
                "input":         user_input,
                "output":        self.guard.get_refusal(),  # Fixed: Use guard refusal helper
                "blocked":       True,
                "block_stage":   "coordinator",
                "block_reason":  reason,
                "category":      category,
                "confidence":    confidence,
                "raw_response":  None,
            }

        # Domain LLM — safe input processed
        raw_response = self.llm.generate(user_input)

        # Guard Agent — post-output validation
        is_safe, cleaned, guard_reason = self.guard.validate(raw_response)

        if not is_safe:
            return {
                "pipeline":      "coordinator",
                "input":         user_input,
                "output":        self.guard.get_refusal(),
                "blocked":       True,
                "block_stage":   "guard",
                "block_reason":  guard_reason,
                "category":      category,
                "confidence":    confidence,
                "raw_response":  raw_response,
            }

        # Buffer-1
        buffered = self._buffer(cleaned, stage=1)

        # Buffer-2
        buffered = self._buffer(buffered, stage=2)

        return {
            "pipeline":     "coordinator",
            "input":        user_input,
            "output":       buffered,
            "blocked":      False,
            "block_stage":  None,
            "block_reason": None,
            "category":     category,
            "confidence":   confidence,
            "raw_response": raw_response,
        }