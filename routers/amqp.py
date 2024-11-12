from datetime import datetime

from faststream.rabbit import RabbitBroker
from faststream.rabbit.fastapi import Logger, RabbitRouter  # type: ignore
from faststream.rabbit.schemas.queue import RabbitQueue
from pydantic import BaseModel
from schema import AMQPPayload, ResponseModel
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue
from utils.credentials import load_rabbitmq_credentials
from utils.persist_payload import PersistPayload
from utils.traceloop import init_tracing
from worker.tasks import query_data_sources
from worker.utils.fire_event import job_send

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds["url"])


class Payload(BaseModel):
    event: str
    date: datetime | str
    content: AMQPPayload


@router.subscriber(queue=RabbitQueue(name=Queue.HIVEMIND, durable=True))
async def ask(payload: Payload, logger: Logger):
    if payload.event == Event.HIVEMIND.QUESTION_RECEIVED:
        try:
            question = payload.content.question.message
            community_id = payload.content.communityId
            init_tracing()
            logger.info(f"COMMUNITY_ID: {community_id} Received job")
            response = query_data_sources(community_id=community_id, query=question)
            logger.info(f"COMMUNITY_ID: {community_id} Job finished")

            response_payload = AMQPPayload(
                communityId=community_id,
                route=payload.content.route,
                question=payload.content.question,
                response=ResponseModel(message=response),
                metadata=payload.content.metadata,
            )
            # dumping the whole payload of question & answer to db
            persister = PersistPayload()
            persister.persist_amqp(response_payload)

            job_send(
                event=payload.content.route.destination.event,
                queue_name=payload.content.route.destination.queue,
                content=response_payload.model_dump(),
            )
        except Exception as e:
            logger.exception(f"Errors While processing job! {e}")
    else:
        logger.error(
            f"No such `{payload.event}` event available for {Queue.HIVEMIND} queue!"
        )
