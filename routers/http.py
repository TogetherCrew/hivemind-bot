from celery.result import AsyncResult
from fastapi import APIRouter
from pydantic import BaseModel
from worker.tasks import ask_question_auto_search


class Payload(BaseModel):
    question: str
    response: str | None = None
    community_id: str
    # bot_given_info: dict[str, Any]


router = APIRouter()


@router.post("/ask")
async def ask(payload: Payload):
    query = payload.question
    community_id = payload.community_id
    task = ask_question_auto_search.delay(
        community_id=community_id,
        query=query,
    )
    return {"id": task.id}


@router.get("/status")
async def status(task_id: str):
    task = AsyncResult(task_id)
    return {"id": task.id, "status": task.status, "result": task.result}
