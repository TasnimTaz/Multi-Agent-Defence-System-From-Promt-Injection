# pipelines/macd_pipeline_v2.py

from concurrent.futures import ThreadPoolExecutor
import time

from agents.macd_pattern_expert import PatternExpertAgent
from agents.macd_intent_expert import IntentExpertAgent
from agents.macd_category_expert import CategoryExpertAgent
from agents.macd_judge import MACDJudgeAgent
from agents.domain_llm import DomainLLM
from agents.guard import GuardAgent
from config import (
    MACD_V2_PATTERN_MODEL,
    MACD_V2_INTENT_MODEL,
    MACD_V2_CATEGORY_MODEL,
    MACD_V2_JUDGE_MODEL,
)


class MACDPipelineV2:
    """
    MACD-v2 — Diverse-Model Ensemble variant.

    MACD-v1 (MACDPipeline) এর প্রতিটা agent একই DEFENSE_MODEL শেয়ার করে, তাই
    architecture-only effect measure করা যায় কিন্তু agent-দের মধ্যে সত্যিকারের
    "expert diversity" থাকে না (একই model-কে চারবার জিজ্ঞেস করা)।

    MACD-v2 তে প্রতিটা agent ইচ্ছাকৃতভাবে ভিন্ন model ব্যবহার করে:
        - Pattern/Syntax Expert  → দ্রুত, ছোট model (llama-3.1-8b-instant)
        - Intent/Action Expert   → reasoning model (qwen/qwen3-32b)
        - Attack-Category Expert → ভিন্ন lineage model (openai/gpt-oss-120b)
        - Judge                 → সবচেয়ে নির্ভরযোগ্য model (llama-3.3-70b-versatile)

    লক্ষ্য: architecture (multi-agent decomposition) + genuine model diversity
    একসাথে ব্যবহার করলে detection ensemble সত্যিই উন্নত হয় কিনা সেটা টেস্ট করা,
    MACD-v1 (same-model) এর তুলনায়।
    """

    def __init__(self, domain_llm: DomainLLM):
        self.pattern_expert = PatternExpertAgent(model=MACD_V2_PATTERN_MODEL)
        self.intent_expert = IntentExpertAgent(model=MACD_V2_INTENT_MODEL)
        self.category_expert = CategoryExpertAgent(model=MACD_V2_CATEGORY_MODEL)
        self.judge = MACDJudgeAgent(model=MACD_V2_JUDGE_MODEL)
        self.llm = domain_llm
        self.guard = GuardAgent()

    def _run_experts_parallel(self, user_input: str) -> tuple[dict, dict, dict]:
        """
        তিনটা ভিন্ন model-ভিত্তিক expert agent independently চলে। ভিন্ন model/endpoint
        হওয়ায় rate-limit বাজেট আলাদা আলাদা, তাই burst risk MACD-v1 এর তুলনায় কম —
        তাও সতর্কতার জন্য ছোট stagger রাখা হলো।
        """
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_pattern = executor.submit(self.pattern_expert.analyze, user_input)
            time.sleep(0.3)
            f_intent = executor.submit(self.intent_expert.analyze, user_input)
            time.sleep(0.3)
            f_category = executor.submit(self.category_expert.analyze, user_input)

            pattern_verdict = f_pattern.result()
            intent_verdict = f_intent.result()
            category_verdict = f_category.result()

        return pattern_verdict, intent_verdict, category_verdict

    def _buffer(self, response: str, stage: int) -> str:
        """Coordinator/MACD-v1 এর সাথে সামঞ্জস্যপূর্ণ 3-bullet rule buffer।"""
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
        # Phase 3: তিনটা ভিন্ন-model expert agent parallel-এ চলবে
        pattern_verdict, intent_verdict, category_verdict = self._run_experts_parallel(user_input)

        # Agent 4: Judge (নির্ভরযোগ্য model) — তিনটা verdict synthesize করে চূড়ান্ত সিদ্ধান্ত
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
                "pipeline": "macd_v2",
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
                "pipeline": "macd_v2",
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
            "pipeline": "macd_v2",
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