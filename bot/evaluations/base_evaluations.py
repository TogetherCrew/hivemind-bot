from abc import ABC, abstractmethod
from typing import Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .schema import EvaluationResult


class BaseEvaluation(ABC):
    def __init__(
        self, model: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.0
    ):
        """
        Initialize the base evaluation class.

        Parameters
        ----------
        model : str, optional
            The OpenAI model to use for evaluations, by default "gpt-4.1-mini-2025-04-14"
        temperature : float, optional
            Temperature for model responses (0.0 for deterministic), by default 0.0
        """
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _get_llm_response(self, messages: list[dict[str, str]]) -> str:
        """
        Get a response from the LLM with retry logic.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries with role and content

        Returns
        -------
        str
            The LLM's response content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            raise

    @abstractmethod
    async def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Abstract method that must be implemented by all evaluation classes.

        Returns
        -------
        EvaluationResult
            Evaluation results as a Pydantic model
        """
        pass

    def _create_system_message(self, evaluation_type: str) -> dict[str, str]:
        """
        Create a system message for the evaluation.

        Parameters
        ----------
        evaluation_type : str
            Type of evaluation being performed

        Returns
        -------
        dict[str, str]
            System message dictionary
        """
        return {
            "role": "system",
            "content": f"You are an expert evaluator performing {evaluation_type}. "
            f"Provide clear, objective evaluations based on the given criteria.",
        }
