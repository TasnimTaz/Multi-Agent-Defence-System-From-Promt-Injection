import sys
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector

# রুট পাথ সেটিংস
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agents.macd_pattern_expert import PatternExpertAgent
from agents.macd_intent_expert import IntentExpertAgent
from agents.macd_category_expert import CategoryExpertAgent
from agents.macd_judge import MACDJudgeAgent
from config import MACD_V2_PATTERN_MODEL, MACD_V2_INTENT_MODEL, MACD_V2_CATEGORY_MODEL, MACD_V2_JUDGE_MODEL

logger = logging.getLogger("macd_defense")


class MACDPromptInjectionDetector(PromptInjectionDetector):
    def __init__(self, mode: str = "full_conversation", raise_on_injection: bool = True):
        super().__init__(mode=mode, raise_on_injection=raise_on_injection)
        self.pattern_expert = PatternExpertAgent(model=MACD_V2_PATTERN_MODEL)
        self.intent_expert = IntentExpertAgent(model=MACD_V2_INTENT_MODEL)
        self.category_expert = CategoryExpertAgent(model=MACD_V2_CATEGORY_MODEL)
        self.judge = MACDJudgeAgent(model=MACD_V2_JUDGE_MODEL)

    def _execute_with_retry(self, agent_fn, *args, max_retries: int = 3, initial_delay: float = 2.0):
        delay = initial_delay
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return agent_fn(*args)
            except Exception as e:
                last_exc = e
                if ("429" in str(e) or "Too Many Requests" in str(e)) and attempt < max_retries - 1:
                    logger.warning(
                        "[MACD Retry] %s hit transient error, retrying in %.1fs (attempt %d/%d)",
                        getattr(agent_fn, "__qualname__", agent_fn), delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def detect(self, tool_output: str) -> tuple[bool, float]:
        clean_output = (tool_output or "").strip()
        if len(clean_output) > 6000:
            clean_output = clean_output[:2000] + "\n...\n" + clean_output[-4000:]
        if not clean_output or clean_output in ["...", "None", "{}", "[]"] or len(clean_output) < 5:
            return False, 1.0

        try:
            # ============================================================
            # IMPORTANT FIX: each expert is dispatched to the thread pool
            # EXACTLY ONCE, and its result is retrieved via .result() below.
            # The previous version submitted these 3 calls to the executor
            # AND THEN called the same 3 analyze() methods again
            # synchronously without ever reading the futures' results --
            # meaning every detect() call was silently making 6 LLM calls
            # instead of 3 (3 wasted in background threads, 3 duplicated
            # synchronously), which was the primary driver of the 429
            # rate-limit storm seen in benchmark runs. Do NOT re-add a
            # second synchronous call block here.
            # ============================================================
            with ThreadPoolExecutor(max_workers=3) as executor:
                f_pattern = executor.submit(self._execute_with_retry, self.pattern_expert.analyze, clean_output)
                f_intent = executor.submit(self._execute_with_retry, self.intent_expert.analyze, clean_output)
                f_category = executor.submit(self._execute_with_retry, self.category_expert.analyze, clean_output)

                pattern_verdict = f_pattern.result()
                intent_verdict = f_intent.result()
                category_verdict = f_category.result()

            logger.debug("[MACD] pattern=%s intent=%s category=%s", pattern_verdict, intent_verdict, category_verdict)

            is_safe, category, reason, confidence = self._execute_with_retry(
                self.judge.synthesize, clean_output, pattern_verdict, intent_verdict, category_verdict
            )

            is_injection = not is_safe
            if is_injection:
                logger.warning("[MACD] Security Alert: Injection detected! Reason: %s", reason)
            return is_injection, confidence

        except Exception as e:
            logger.error("[MACD-v2 Defense Error]: %s", e)
            # Fail-secure: real agent-chain crash -> treat as injection (True).
            # With raise_on_injection=True this triggers a redact/abort via the
            # base class, which is the safer default when the detector itself
            # is malfunctioning.
            return True, 0.5