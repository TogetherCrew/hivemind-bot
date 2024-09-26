from datetime import datetime

from faststream.rabbit.fastapi import Logger, RabbitRouter  # type: ignore
from faststream.rabbit.schemas.queue import RabbitQueue
from pydantic import BaseModel
from schema import PayloadModel, InputModel, OutputModel
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue
from utils.credentials import load_rabbitmq_credentials
from worker.tasks import query_data_sources

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds["url"])


class Payload(BaseModel):
    event: str
    date: datetime | str
    content: PayloadModel


@router.subscriber(queue=RabbitQueue(name=Queue.HIVEMIND, durable=True))
@router.publisher(queue=RabbitQueue(Queue.DISCORD_BOT, durable=True))
async def ask(payload: Payload, logger: Logger):
    if payload.event == Event.HIVEMIND.INTERACTION_CREATED:
        try:
            question = payload.content.input.message
            community_id = payload.content.input.community_id

            logger.info(f"COMMUNITY_ID: {community_id} Received job")
            response = query_data_sources(community_id=community_id, query=question)
            logger.info(f"COMMUNITY_ID: {community_id} Job finished")

            response_payload = PayloadModel(
                input=InputModel(message=response, community_id=community_id),
                output=OutputModel(destination=payload.content.output.destination),
                metadata=payload.content.metadata,
                session_id=payload.content.session_id,
            )
            result = Payload(
                event=payload.content.output.destination,
                date=str(datetime.now()),
                content=response_payload.model_dump(),
            )
            return result
        except Exception as e:
            logger.error(f"Errors While processing job! {e}")
    else:
        logger.error(
            f"No more event available for {Queue.HIVEMIND} queue! "
            f"Received event: `{payload.event}`"
        )
