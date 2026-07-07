# agents/groq_utils.py

import re
import time
import random


def safe_completion(client, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
    """
    Wrapper around client.chat.completions.create() that retries on
    HTTP 429 (rate limit) with exponential backoff + jitter.

    Groq error bodies for 429 usually look like:
        Error code: 429 - {'error': {'message': 'Rate limit reached for model
        `qwen/qwen3.6-27b` ... Please try again in 1.234s ...'}}

    We try to parse the "try again in X s" hint from the message; if we
    can't find it, we fall back to exponential backoff.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate_limit = "429" in msg or "rate limit" in msg.lower()

            if not is_rate_limit:
                # Not a rate-limit error — no point retrying, raise immediately
                raise

            # Try to parse Groq's suggested wait time, e.g. "try again in 1.234s"
            wait_match = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
            if wait_match:
                wait_time = float(wait_match.group(1)) + 0.5  # small safety margin
            else:
                # Exponential backoff with jitter: 2s, 4s, 8s, 16s, 32s...
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)

            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                # Out of retries — let caller's except block handle it
                raise last_err