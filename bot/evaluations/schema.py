from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """
    Base model for evaluation results.

    Attributes
    ----------
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    question: str
    answer: str


class AnswerRelevanceSuccess(EvaluationResult):
    """
    Model for successful answer relevance evaluation results.

    Attributes
    ----------
    score : int
        Relevance score from 1-10
    explanation : str
        Brief explanation of the score
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    score: int
    explanation: str


class AnswerRelevanceError(EvaluationResult):
    """
    Model for failed answer relevance evaluation results.

    Attributes
    ----------
    error : str
        Error message describing what went wrong
    raw_response : str
        The raw response from the LLM that couldn't be parsed
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    error: str
    raw_response: str


class AnswerConfidenceSuccess(EvaluationResult):
    """
    Model for successful answer confidence evaluation results.

    Attributes
    ----------
    score : int
        Confidence score from 1-10
    explanation : str
        Brief explanation of the score
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    score: int
    explanation: str


class AnswerConfidenceError(EvaluationResult):
    """
    Model for failed answer confidence evaluation results.

    Attributes
    ----------
    error : str
        Error message describing what went wrong
    raw_response : str
        The raw response from the LLM that couldn't be parsed
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    error: str
    raw_response: str


class QuestionAnswerCoverageSuccess(EvaluationResult):
    """
    Model for successful question answer coverage evaluation results.

    Attributes
    ----------
    answered : bool
        Whether the question was actually answered (True/False)
    score : int
        Score from 1-10 indicating how well the question was answered
    explanation : str
        Brief explanation of the evaluation
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    answered: bool
    score: int
    explanation: str


class QuestionAnswerCoverageError(EvaluationResult):
    """
    Model for failed question answer coverage evaluation results.

    Attributes
    ----------
    error : str
        Error message describing what went wrong
    raw_response : str
        The raw response from the LLM that couldn't be parsed
    question : str
        The question that was asked
    answer : str
        The answer that was evaluated
    """

    error: str
    raw_response: str
