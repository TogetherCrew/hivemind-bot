from datetime import datetime

from pydantic import BaseModel
from faststream.rabbit.fastapi import RabbitRouter, Logger  # type: ignore
from faststream.rabbit.schemas.queue import RabbitQueue
from utils.credentials import load_rabbitmq_credentials
from tc_messageBroker.rabbit_mq.queue import Queue
from tc_messageBroker.rabbit_mq.event import Event
from worker.tasks import query_data_sources

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds["url"])


class Content(BaseModel):
    question: str
    community_id: str


class Payload(BaseModel):
    event: str
    date: datetime | str
    content: Content | dict


@router.subscriber(queue=RabbitQueue(name=Queue.HIVEMIND, durable=True))
@router.publisher(queue=RabbitQueue(Queue.DISCORD_BOT, durable=True))
async def ask(payload: Payload, logger: Logger):
    if payload.event == Event.HIVEMIND.INTERACTION_CREATED:
        question = payload.content.question
        community_id = payload.content.community_id

        logger.info(f"COMMUNITY_ID: {community_id} Received job")
        response = query_data_sources(community_id=community_id, query=question)
        logger.info(f"COMMUNITY_ID: {community_id} Job finished")

        response_payload = Payload(
            event=Event.DISCORD_BOT.INTERACTION_RESPONSE.EDIT,
            date=str(datetime.now()),
            content={"response": response},
        )
    else:
        raise NotImplementedError(
            f"No more event available for {Queue.HIVEMIND} queue! "
            f"Received event: `{payload.event}`"
        )
    return response_payload
