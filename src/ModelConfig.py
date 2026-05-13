from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 512

    @property
    def full_model_name(self) -> str:
        return self.model_name

