# pipelines/macd_pipeline.py

from concurrent.futures import ThreadPoolExecutor
import time

from agents.macd_pattern_expert import PatternExpertAgent
from agents.macd_intent_expert import IntentExpertAgent
from agents.macd_category_expert import CategoryExpertAgent
from agents.macd_judge import MACDJudgeAgent
from agents.domain_llm import DomainLLM
from agents.guard import GuardAgent


class MACDPipeline:
    """
    MACD-inspired Multi-Agent Pipeline (adapted from Li et al., "MACD: Multi-Agent
    Collaborative Approach for Cybersecurity Defense Strategy Generation",
    Information 2026, 17, 370 — re-mapped from "defense strategy generation" to
    "prompt injection detection", alongside এই প্রজেক্টের Chain ও Coordinator
    পাইপলাইনের পাশে একটা নতুন তৃতীয় বিকল্প হিসেবে:

    User Input
        → Phase A: KB Retrieval (RAG-lite, inside Agent 3)
        → [Agent 1: Pattern/Syntax Expert] ─┐
        → [Agent 2: Intent/Action Expert]   ─┼─ parallel ──▶ [Agent 4: Judge/Coordinator]
        → [Agent 3: Attack-Category Expert] ─┘
            → [Attack] → Safe Refusal + Logger
            → [Safe]   → Domain LLM → Guard Agent → Buffer-1 → Buffer-2 → Output + Logger
    """

    def __init__(self, domain_llm: DomainLLM):
        self.pattern_expert = PatternExpertAgent()
        self.intent_expert = IntentExpertAgent()
        self.category_expert = CategoryExpertAgent()
        self.judge = MACDJudgeAgent()
        self.llm = domain_llm
        self.guard = GuardAgent()

    def _run_experts_parallel(self, user_input: str) -> tuple[dict, dict, dict]:
        """
        Paper Sec. 4.4: তিনটা expert agent independently চলে, কিন্তু rate-limit
        burst এড়াতে submit-এর মাঝে ছোট stagger রাখা হয়েছে (একসাথে ৩টা call
        fire হলে TPM/RPM burst হয়ে 429 আসার সম্ভাবনা বেশি থাকে)।
        """
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_pattern = executor.submit(self.pattern_expert.analyze, user_input)
            time.sleep(0.5)
            f_intent = executor.submit(self.intent_expert.analyze, user_input)
            time.sleep(0.5)
            f_category = executor.submit(self.category_expert.analyze, user_input)

            pattern_verdict = f_pattern.result()
            intent_verdict = f_intent.result()
            category_verdict = f_category.result()

        return pattern_verdict, intent_verdict, category_verdict

    def _buffer(self, response: str, stage: int) -> str:
        """Paper Fig. 3-স্টাইল Buffer — 3-bullet rule enforce করা (Coordinator পাইপলাইনের সাথে সামঞ্জস্যপূর্ণ)।"""
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
        return '\n'.join(result)

    def run(self, user_input: str) -> dict:
        # Phase 3: তিনটা expert agent parallel-এ চলবে
        pattern_verdict, intent_verdict, category_verdict = self._run_experts_parallel(user_input)

        # Agent 4: Judge — তিনটা verdict synthesize করে চূড়ান্ত সিদ্ধান্ত
        is_safe, category, reason, confidence = self.judge.synthesize(
            user_input, pattern_verdict, intent_verdict, category_verdict
        )

        agent_verdicts = {
            "pattern_expert": pattern_verdict,
            "intent_expert": intent_verdict,
            "category_expert": category_verdict,
        }

        if not is_safe:
            return {
                "pipeline": "macd",
                "input": user_input,
                "output": self.guard.get_refusal(),
                "blocked": True,
                "block_stage": "macd_judge",
                "block_reason": reason,
                "category": category,
                "confidence": confidence,
                "raw_response": None,
                "agent_verdicts": agent_verdicts,
            }

        # Domain LLM — safe input processed
        raw_response = self.llm.generate(user_input)

        # Guard Agent — post-output validation
        is_safe, cleaned, guard_reason = self.guard.validate(raw_response)

        if not is_safe:
            return {
                "pipeline": "macd",
                "input": user_input,
                "output": self.guard.get_refusal(),
                "blocked": True,
                "block_stage": "guard",
                "block_reason": guard_reason,
                "category": category,
                "confidence": confidence,
                "raw_response": raw_response,
                "agent_verdicts": agent_verdicts,
            }

        # Buffer-1, Buffer-2
        buffered = self._buffer(cleaned, stage=1)
        buffered = self._buffer(buffered, stage=2)

        return {
            "pipeline": "macd",
            "input": user_input,
            "output": buffered,
            "blocked": False,
            "block_stage": None,
            "block_reason": None,
            "category": category,
            "confidence": confidence,
            "raw_response": raw_response,
            "agent_verdicts": agent_verdicts,
        }