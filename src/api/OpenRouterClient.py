from src.api.OpenRouterError import OpenRouterError
from src.ModelConfig  import ModelConfig
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class OpenRouterClient:

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, logger, timeout: int = 120) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logger

    @retry(
        retry=retry_if_exception_type((requests.RequestException, OpenRouterError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate_response(self, prompt: str, model_config: ModelConfig) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_config.full_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
        }

        self.logger.info("Calling model=%s", model_config.full_model_name,)
        response = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code != 200:
            raise OpenRouterError(f"API Error {response.status_code}: {response.text}")

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise OpenRouterError(f"Unexpected response structure: {data}") from exc

