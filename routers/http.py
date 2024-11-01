from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel
from schema import HTTPPayload, QuestionModel, ResponseModel
from utils.persist_payload import PersistPayload
from worker.tasks import ask_question_auto_search


class RequestPayload(BaseModel):
    question: QuestionModel
    communityId: str


router = APIRouter()


@router.post("/ask")
async def ask(payload: RequestPayload):
    query = payload.question.message
    community_id = payload.communityId
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
async def status(task_id: str):
    task = AsyncResult(task_id)

    # persisting the data updates in db
    persister = PersistPayload()

    http_payload = HTTPPayload(
        communityId=task.result["community_id"],
        question=QuestionModel(message=task.result["question"]),
        response=ResponseModel(message=task.result["response"]),
        taskId=task.id,
    )
    persister.persist_http(http_payload, update=True)

    return {"id": task.id, "status": task.status, "result": task.result}