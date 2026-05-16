from src.ModelConfig import ModelConfig
from src.api.OpenRouterClient import OpenRouterClient
from src.QuestionProcessor import QuestionProcessor
import logging
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("Batch_LLM_Processor")


def build_models() -> List[ModelConfig]:
    return [
        ModelConfig(
            provider="openai",
            model_name="openai/gpt-4.1-mini",
            temperature=0.2,
        ),
        ModelConfig(
            provider="deepseek",
            model_name="deepseek/deepseek-chat",
            temperature=0.2,
        ),
        ModelConfig(
            provider="google",
            model_name="~google/gemini-pro-latest",
            temperature=0.2,
        ),
        ModelConfig(
            provider="meta",
            model_name="nousresearch/hermes-3-llama-3.1-405b",
            temperature=0.2,
        ),
    ]


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is missing.")

    client = OpenRouterClient(api_key=api_key, logger=LOGGER)
    processor = QuestionProcessor(
        csv_path="benchmarks\FinanceQA.csv",
        client=client,
        models=build_models(),
        logger=LOGGER,
        save_every_n_rows=1,
        delay_between_requests=5,
    )
    processor.process()


if __name__ == "__main__":
    main()