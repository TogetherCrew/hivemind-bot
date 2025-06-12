from typing import Union
from .base_evaluations import BaseEvaluation
from .schema import QuestionAnswerCoverageSuccess, QuestionAnswerCoverageError


class QuestionAnswerCoverageEvaluation(BaseEvaluation):
    def __init__(
        self, model: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.0
    ):
        """
        Initialize the question answer coverage evaluation class.

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
    ) -> Union[QuestionAnswerCoverageSuccess, QuestionAnswerCoverageError]:
        """
        Evaluate whether a question was actually answered.

        Parameters
        ----------
        question : str
            The question that was asked
        answer : str
            The answer to evaluate

        Returns
        -------
        Union[QuestionAnswerCoverageSuccess, QuestionAnswerCoverageError]
            Pydantic model containing whether the question was answered, score, and explanation or error details
        """
        system_message = self._create_system_message("question answered evaluation")

        evaluation_prompt = f"""Please evaluate whether the question was actually answered and how well it was answered.
        
        Consider the following criteria:
        1. Does the answer provide specific information that addresses the question?
        2. Is the answer substantive (not just "I don't know" or "No information available")?
        3. Does the answer contain relevant details or facts related to the question?
        4. Is the answer complete enough to satisfy the question being asked?
        5. Does the answer avoid deflecting or redirecting without providing information?

        Examples of answers that would be considered "not answered":
        - "I don't have information about that"
        - "I cannot answer this question"
        - "No relevant information found"
        - Completely off-topic responses
        - Empty or very vague responses

        Question: {question}
        Answer: {answer}

        Provide your evaluation in the following format:
        Answered: [True/False]
        Score: [1-10]
        Explanation: [Brief explanation of whether and how well the question was answered]
        """

        messages = [system_message, {"role": "user", "content": evaluation_prompt}]

        response = await self._get_llm_response(messages)

        # Parse the response to extract answered status, score and explanation
        try:
            answered_line = [
                line for line in response.split("\n") if line.startswith("Answered:")
            ][0]
            score_line = [
                line for line in response.split("\n") if line.startswith("Score:")
            ][0]
            explanation_line = [
                line for line in response.split("\n") if line.startswith("Explanation:")
            ][0]

            answered = answered_line.split(":")[1].strip().lower() == "true"
            score = int(score_line.split(":")[1].strip())
            explanation = explanation_line.split(":")[1].strip()

            return QuestionAnswerCoverageSuccess(
                answered=answered,
                score=score,
                explanation=explanation,
                question=question,
                answer=answer,
            )
        except (IndexError, ValueError) as e:
            return QuestionAnswerCoverageError(
                error=f"Failed to parse evaluation response: {str(e)}",
                raw_response=response,
                question=question,
                answer=answer,
            ) 