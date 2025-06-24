from typing import Union
from .base_evaluations import BaseEvaluation
from .schema import AnswerConfidenceSuccess, AnswerConfidenceError


class AnswerConfidenceEvaluation(BaseEvaluation):
    def __init__(
        self, model: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.0
    ):
        """
        Initialize the answer confidence evaluation class.

        Parameters
        ----------
        model : str, optional
            The OpenAI model to use for evaluations, by default "gpt-4.1-mini-2025-04-14"
        temperature : float, optional
            Temperature for model responses, by default 0.0
        """
        super().__init__(model, temperature)

    async def evaluate(
        self, question: str, answer: str
    ) -> Union[AnswerConfidenceSuccess, AnswerConfidenceError]:
        """
        Evaluate the confidence level of an answer to a given question.

        Parameters
        ----------
        question : str
            The question that was asked
        answer : str
            The answer to evaluate

        Returns
        -------
        Union[AnswerConfidenceSuccess, AnswerConfidenceError]
            Pydantic model containing the confidence score and explanation or error details
        """
        system_message = self._create_system_message("answer confidence evaluation")

        evaluation_prompt = f"""Please evaluate how confident the answer appears to be on a scale of 1-10.
        Consider:
        1. Does the answer use confident language or uncertain language?
        2. Are there hedging words like "maybe", "possibly", "I think", etc.?
        3. Does the answer provide specific details or remain vague?
        4. Does the answer acknowledge limitations or uncertainty?
        5. Is the answer backed by specific facts or evidence?

        Question: {question}
        Answer: {answer}

        Provide your evaluation in the following format:
        Score: [1-10]
        Explanation: [Brief explanation of the score]
        """

        messages = [system_message, {"role": "user", "content": evaluation_prompt}]

        response = await self._get_llm_response(messages)

        # Parse the response to extract score and explanation
        try:
            score_line = [
                line for line in response.split("\n") if line.startswith("Score:")
            ][0]
            explanation_line = [
                line for line in response.split("\n") if line.startswith("Explanation:")
            ][0]

            score = int(score_line.split(":")[1].strip())
            explanation = explanation_line.split(":")[1].strip()

            return AnswerConfidenceSuccess(
                score=score, explanation=explanation, question=question, answer=answer
            )
        except (IndexError, ValueError) as e:
            return AnswerConfidenceError(
                error=f"Failed to parse evaluation response: {str(e)}",
                raw_response=response,
                question=question,
                answer=answer,
            )
