from __future__ import annotations

import os
import time
from typing import List
from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError

from .config import Settings
from .types import ChatMessage, ChatResult


class LLMClient:
    """
    Thin wrapper around Azure OpenAI chat completions.
    - Centralizes auth, endpoint, deployment
    - Adds basic retries
    - Returns typed ChatResult with token usage & latency
    - Supports DRY_RUN to skip real API calls during basc local testing
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or Settings.load()

        # set up OA client
        self._client = AzureOpenAI(
            api_key=self.cfg.azure_api_key,
            api_version=self.cfg.api_version,
            azure_endpoint=self.cfg.azure_endpoint,
        )
        self._deployment = self.cfg.azure_deployment

        #safety guard: allow local dry runs for testing
        self._dry_run = os.getenv("DRY_RUN", "0") == "1"

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = 512,
        retries: int = 2,
    ) -> ChatResult:
        """
        Send chat completion request to Azure OpenAI.
        Retries on transient errors. Returns text + token usage + latency.
        """
        if self._dry_run:
            #some simulated response (useful for dev without calling the API)
            fake_text = "[DRY_RUN] Hello from LLMClient"
            return ChatResult(
                text=fake_text,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
                raw={"dry_run": True},
            )

        last_err = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self._deployment,  
                    messages=[m.to_openai_dict() for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0

                choice = resp.choices[0]
                text = (choice.message.content or "").strip()

                usage = getattr(resp, "usage", None)
                prompt_toks = getattr(usage, "prompt_tokens",0) or 0
                completion_toks = getattr(usage,"completion_tokens",0) or 0
                total_toks = getattr(usage, "total_tokens", prompt_toks + completion_toks) or 0

                #convert resp to a plain dict for storage/tracing
                raw_dict = {
                    "id": getattr(resp,"id",None),
                    "model": getattr(resp,"model",None),
                    "choices_len": len(resp.choices),
                    "usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": completion_toks,
                        "total_tokens": total_toks,
                    },
                }

                return ChatResult(
                    text=text,
                    prompt_tokens=prompt_toks,
                    completion_tokens=completion_toks,
                    total_tokens=total_toks,
                    latency_ms=dt_ms,
                    raw=raw_dict,
                )
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_err = e
                #simple backoff
                time.sleep(0.6 * (attempt + 1))
                continue

        # all retries failed
        raise RuntimeError(f"LLM request failed after {retries+1} tries: {last_err}")
