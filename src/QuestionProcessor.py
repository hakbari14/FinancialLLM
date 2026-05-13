from src.api.OpenRouterClient import OpenRouterClient
from src.ModelConfig import ModelConfig
from typing import List

import pandas as pd
import os
import time


class QuestionProcessor:

    def __init__(self, csv_path: str, client: OpenRouterClient, models: List[ModelConfig], logger, save_every_n_rows: int = 5, delay_between_requests: float = 1.0,) -> None:
        self.csv_path = csv_path
        self.client = client
        self.models = models
        self.logger = logger
        self.save_every_n_rows = save_every_n_rows
        self.delay_between_requests = delay_between_requests

    def load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        return df

    def ensure_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for model in self.models:
            col_name = self._response_column_name(model)
            if col_name not in df.columns:
                df[col_name] = ""

        return df

    def process(self) -> None:
        df = self.load_csv()
        df = self.ensure_output_columns(df)

        total_rows = len(df)
        self.logger.info("Processing %s rows", total_rows)

        for idx, row in df.iterrows():
            question = str(row['question']).strip()
            if not question:
                self.logger.warning("Skipping empty question at row %s", idx) 
                continue

            prompt = self.create_prompt_for_question(row)
            self.logger.info("Processing row %s/%s", idx + 1, total_rows,)

            for model in self.models:
                column_name = self._response_column_name(model)
                existing_value = str(row.get(column_name, "")).strip()
                if existing_value:
                    self.logger.info("Skipping existing response for %s",column_name,)
                    continue

                try:
                    response = self.client.generate_response(prompt = prompt, model_config = model)
                    df.at[idx, column_name] = response

                    self.logger.info("Saved response for model=%s", model.full_model_name)

                except Exception as exc:
                    self.logger.exception("Failed processing row=%s model=%s error=%s", idx, model.full_model_name, exc,)
                    df.at[idx, column_name] = (f"ERROR: {str(exc)}")

                time.sleep(self.delay_between_requests)

            if (idx + 1) % self.save_every_n_rows == 0:
                self._safe_save(df)

        self._safe_save(df)
        self.logger.info("Processing completed successfully.")

    def create_prompt_for_question(self, row) -> None:
        context = str(row['context']).strip()
        question = str(row['question']).strip()
        prompt = f'Context:\n{context}\n\nQuestion: {question}\nProvide a concise answer.' 
        return prompt

    def _safe_save(self, df: pd.DataFrame) -> None:
        temp_path = f"{self.csv_path}.tmp"
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, self.csv_path)
        self.logger.info("CSV saved: %s", self.csv_path)

    @staticmethod
    def _response_column_name(model: ModelConfig) -> str:
        safe_name = (model.full_model_name.replace("/", "_").replace("-", "_").lower())
        return f"response_{safe_name}"
