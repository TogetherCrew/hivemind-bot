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
    answer: str | None = None


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
    raw_response: str | None = None


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
    raw_response: str | None = None


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
    raw_response: str | None = None


class NodeRelevanceSuccess(BaseModel):
    """
    Model for successful node relevance evaluation results.

    Attributes
    ----------
    relevance_score : int
        Relevance score from 1-10
    explanation : str
        Brief explanation of the relevance score
    question : str
        The question that was asked
    node_content : str
        The content of the node that was evaluated
    node_id : str
        The ID of the node that was evaluated
    node_score : float
        The similarity score of the node
    """

    relevance_score: int
    explanation: str
    question: str
    node_content: str
    node_id: str
    node_score: float


class NodeRelevanceError(BaseModel):
    """
    Model for failed node relevance evaluation results.

    Attributes
    ----------
    error : str
        Error message describing what went wrong
    raw_response : str
        The raw response from the LLM that couldn't be parsed
    question : str
        The question that was asked
    node_content : str
        The content of the node that was evaluated
    node_id : str
        The ID of the node that was evaluated
    node_score : float
        The similarity score of the node
    """

    error: str
    raw_response: str | None = None
    question: str
    node_content: str
    node_id: str
    node_score: float


class NodesEvaluationSummary(BaseModel):
    """
    Model for summary of node evaluations.

    Attributes
    ----------
    question : str
        The question that was asked
    total_nodes : int
        Total number of nodes evaluated
    summary_nodes_count : int
        Number of summary nodes evaluated
    raw_nodes_count : int
        Number of raw nodes evaluated
    average_relevance_score : float
        Average relevance score across all successful evaluations
    high_relevance_nodes : int
        Number of nodes with relevance score >= 7
    successful_evaluations : int
        Number of successful evaluations
    failed_evaluations : int
        Number of failed evaluations
    """

    question: str
    total_nodes: int
    summary_nodes_count: int
    raw_nodes_count: int
    average_relevance_score: float
    high_relevance_nodes: int
    successful_evaluations: int
    failed_evaluations: int
