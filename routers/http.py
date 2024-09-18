from worker.tasks import ask_question_auto_search
from celery.result import AsyncResult
from pydantic import BaseModel
from fastapi import APIRouter
from typing import Any


class Payload(BaseModel):
    query: str
    community_id: str
    # bot_given_info: dict[str, Any]


router = APIRouter()


@router.post("/ask")
async def ask(payload: Payload):
    task = ask_question_auto_search.delay(**payload.model_dump())
    return {"id": task.id}


@router.get("/status")
async def status(task_id: str):
    task = AsyncResult(task_id)
    return {"id": task.id, "status": task.status, "result": task.result}
