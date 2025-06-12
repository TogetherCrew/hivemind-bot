import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from bot.evaluations.answer_relevance import AnswerRelevanceEvaluation
from bot.evaluations.answer_confidence import AnswerConfidenceEvaluation
from bot.evaluations.question_answered import QuestionAnswerCoverageEvaluation
from schema import HTTPPayload, QuestionModel, ResponseModel
from services.api_key import validate_token
from starlette.status import HTTP_403_FORBIDDEN
from utils.persist_payload import PersistPayload
from worker.tasks import ask_question_auto_search
from bot.evaluations.schema import AnswerRelevanceSuccess, AnswerConfidenceSuccess, QuestionAnswerCoverageSuccess


class RequestPayload(BaseModel):
    question: QuestionModel


router = APIRouter()


@router.post("/ask")
async def ask(
    payload: RequestPayload,
    community_id: str = Depends(validate_token),
):
    query = payload.question.message
    task = ask_question_auto_search.delay(
        community_id=community_id,
        query=query,
    )
    payload_http = HTTPPayload(
        communityId=community_id,
        question=payload.question,
        taskId=task.id,
    )
    # persisting the payload
    persister = PersistPayload()
    persister.persist_http(payload_http)

    return {"id": task.id}


@router.get("/status")
async def status(
    task_id: str,
    community_id: str = Depends(validate_token),
):
    task = AsyncResult(task_id)
    if task.status == "SUCCESS":
        if task.result["community_id"] != community_id:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Task belongs to another community!",
            )

        eval_result = await AnswerRelevanceEvaluation().evaluate(
            question=task.result["question"], answer=task.result["response"]
        )
        confidence_result = await AnswerConfidenceEvaluation().evaluate(
            question=task.result["question"], answer=task.result["response"]
        )
        coverage_result = await QuestionAnswerCoverageEvaluation().evaluate(
            question=task.result["question"], answer=task.result["response"]
        )
        http_payload = HTTPPayload(
            communityId=community_id,
            question=QuestionModel(message=task.result["question"]),
            response=ResponseModel(message=task.result["response"]),
            taskId=task.id,
            metadata={
                "answer_relevance_score": (
                    eval_result.score
                    if isinstance(eval_result, AnswerRelevanceSuccess)
                    else eval_result.error
                ),
                "answer_relevance_explanation": (
                    eval_result.explanation
                    if isinstance(eval_result, AnswerRelevanceSuccess)
                    else eval_result.error
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
            },
        )

        # persisting the data updates in db
        try:
            persister = PersistPayload()
            persister.persist_http(http_payload, update=True)
        except Exception as e:
            logging.error(f"Failed to persist task result: {e}")

        results = {"id": task.id, "status": task.status, "result": task.result}
    else:
        results = {"id": task.id, "status": task.status}

    return results
