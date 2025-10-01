import logging
from openai import OpenAI

from bot.evaluations.answer_relevance import AnswerRelevanceEvaluation
from bot.evaluations.answer_confidence import AnswerConfidenceEvaluation
from bot.evaluations.question_answered import QuestionAnswerCoverageEvaluation
from bot.evaluations.schema import (
    AnswerRelevanceSuccess,
    AnswerConfidenceSuccess,
    QuestionAnswerCoverageSuccess,
    AnswerRelevanceError,
    AnswerConfidenceError,
    QuestionAnswerCoverageError,
)
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
from schema import RouteModel, RouteModelPayload, QuestionModel, ResponseModel
from utils.persist_payload import PersistPayload
from utils.globals import NO_ANSWER_REFERENCE, NO_ANSWER_REFERENCE_PLACEHOLDER


async def general_llm_tool(payload: HivemindQueryPayload):
    """
    Answer using only the LLM's own knowledge (no retrieval), while mirroring
    rag_tool's evaluation and persistence behavior.
    """
    # Generate answer directly from the LLM
    try:
        client = OpenAI()
        model_name = "gpt-4o-mini"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Rely solely on your own knowledge. "
                    "Do not fabricate citations or sources. Provide concise, clear, and less than a paragraph answers. "
                    "Never provide suggestions or ask for clarifications. "
                    f"In case you didn't know the answer, just say '{NO_ANSWER_REFERENCE_PLACEHOLDER}'."
                ),
            },
            {"role": "user", "content": payload.query},
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
        )
        response: str | None = completion.choices[0].message.content if completion else None
        if response == NO_ANSWER_REFERENCE_PLACEHOLDER:
            response = NO_ANSWER_REFERENCE
    except Exception as e:
        logging.exception(f"LLM generation failed. Exception: {str(e)}")
        response = None

    # There are no retrieval references in this mode
    references: list = []

    # Run evaluations similar to rag_tool
    if response:
        relevancy_result = await AnswerRelevanceEvaluation().evaluate(
            question=payload.query, answer=response
        )
        confidence_result = await AnswerConfidenceEvaluation().evaluate(
            question=payload.query, answer=response
        )
        coverage_result = await QuestionAnswerCoverageEvaluation().evaluate(
            question=payload.query, answer=response
        )
    else:
        relevancy_result = AnswerRelevanceError(
            error="No response from the LLM",
            question=payload.query,
            answer=None,
        )
        confidence_result = AnswerConfidenceError(
            error="No response from the LLM",
            question=payload.query,
            answer=None,
        )
        coverage_result = QuestionAnswerCoverageError(
            error="No response from the LLM",
            question=payload.query,
            answer=None,
        )

    evaluation_metadata = {
        "answer_relevance_score": (
            relevancy_result.score
            if isinstance(relevancy_result, AnswerRelevanceSuccess)
            else relevancy_result.error
        ),
        "answer_relevance_explanation": (
            relevancy_result.explanation
            if isinstance(relevancy_result, AnswerRelevanceSuccess)
            else relevancy_result.error
        ),
        "answer_confidence_score": (
            confidence_result.score
            if isinstance(confidence_result, AnswerConfidenceSuccess)
            else confidence_result.error
        ),
        "answer_confidence_explanation": (
            confidence_result.explanation
            if isinstance(confidence_result, AnswerConfidenceSuccess)
            else confidence_result.error
        ),
        "answer_coverage_answered": (
            coverage_result.answered
            if isinstance(coverage_result, QuestionAnswerCoverageSuccess)
            else False
        ),
        "answer_coverage_score": (
            coverage_result.score
            if isinstance(coverage_result, QuestionAnswerCoverageSuccess)
            else coverage_result.error
        ),
        "answer_coverage_explanation": (
            coverage_result.explanation
            if isinstance(coverage_result, QuestionAnswerCoverageSuccess)
            else coverage_result.error
        ),
        # Explicitly indicate no retrieval context was used
        "retrieval_used": False,
        "references_count": 0,
    }

    # Prepare response payload (no references appended)
    response_payload = RouteModelPayload(
        communityId=payload.community_id,
        route=RouteModel(source="temporal", destination=None),
        question=QuestionModel(message=payload.query),
        response=ResponseModel(message=response or ""),
        metadata=evaluation_metadata,
    )

    # Persist like rag_tool (insert or update by workflow_id)
    workflow_id = getattr(payload, "workflow_id", None)
    PersistPayload().persist_payload(response_payload, workflow_id=workflow_id)

    # Optional: apply the same skipping logic as rag_tool when auto-answering
    if (
        isinstance(coverage_result, QuestionAnswerCoverageSuccess)
        and coverage_result.score < 3
        and payload.enable_answer_skipping
    ):
        logging.warning(
            f"Answer coverage score is less than 3, skipping answer: {coverage_result.score}"
        )
        return None, []

    return response, references


