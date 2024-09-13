from celery.result import AsyncResult
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
from worker.tasks import ask_question_auto_search

class Payload(BaseModel):
    question: str
    community_id: str
    bot_given_info: dict[str, Any]

app = FastAPI()

@app.post("/ask")
async def ask(payload: Payload):
    task = ask_question_auto_search.delay(**payload.model_dump())
    return { "id": task.id }

@app.get("/status")
async def status(task_id: str):
    task = AsyncResult(task_id)
    return { "id": task.id, "status": task.status, "result": task.result }
    