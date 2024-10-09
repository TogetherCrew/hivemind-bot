from datetime import datetime

from faststream.rabbit.fastapi import Logger, RabbitRouter  # type: ignore
from faststream.rabbit import RabbitBroker
from faststream.rabbit.schemas.queue import RabbitQueue
from pydantic import BaseModel
from schema import AMQPPayload, ResponseModel
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue
from utils.credentials import load_rabbitmq_credentials
from utils.persist_payload import PersistPayload
from worker.tasks import query_data_sources

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds["url"])


class Payload(BaseModel):
    event: str
    date: datetime | str
    content: AMQPPayload


@router.subscriber(queue=RabbitQueue(name=Queue.HIVEMIND, durable=True))
async def ask(payload: Payload, logger: Logger):
    if payload.event == Event.HIVEMIND.INTERACTION_CREATED:
        try:
            question = payload.content.question.message
            community_id = payload.content.communityId

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

            result = Payload(
                event=payload.content.route.destination.event,
                date=str(datetime.now()),
                content=response_payload.model_dump(),
            )
            async with RabbitBroker(url=rabbitmq_creds["url"]) as broker:
                await broker.publish(
                    message=result, queue=payload.content.route.destination.queue
                )
        except Exception as e:
            logger.exception(f"Errors While processing job! {e}")
    else:
        logger.error(
            f"No such `{payload.event}` event available for {Queue.HIVEMIND} queue!"
        )
