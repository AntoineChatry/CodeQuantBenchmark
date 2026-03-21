"""Moteur d'inférence via API OpenAI-compatible (llama.cpp server)."""

import time
from typing import Any

from openai import OpenAI

from src.inference.base import InferenceEngine, InferenceResult
from src.utils.logging import setup_logger

logger = setup_logger("inference_openai")


class OpenAICompatEngine(InferenceEngine):
    def __init__(self, api_base: str, api_key: str = "not-needed", **kwargs: Any):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name: str = ""
        self.client: OpenAI | None = None
        self.extra_params = kwargs
        self._total_latency_ms: float = 0.0
        self._total_requests: int = 0

    def load(self, model_path: str) -> None:
        self.model_name = model_path
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        logger.info(f"Connected to {self.api_base}, model={model_path}")

    def generate(self, prompt: str, max_tokens: int = 512) -> InferenceResult:
        if self.client is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        start = time.perf_counter()
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self.extra_params.get("temperature", 0.1),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._total_latency_ms += elapsed_ms
        self._total_requests += 1

        choice = response.choices[0]
        usage = response.usage

        return InferenceResult(
            text=choice.text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
        )

    def unload(self) -> None:
        self.client = None
        self.model_name = ""
        logger.info("Engine unloaded")

    def get_metrics(self) -> dict[str, float]:
        avg = (
            self._total_latency_ms / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "total_latency_ms": round(self._total_latency_ms, 2),
            "avg_latency_ms": round(avg, 2),
        }
