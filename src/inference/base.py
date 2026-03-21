"""Story 3.2 : Interface abstraite du moteur d'inférence."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InferenceResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class InferenceEngine(ABC):
    @abstractmethod
    def load(self, model_path: str) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> InferenceResult:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        pass
