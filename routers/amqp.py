from fastapi import Depends
from pydantic import BaseModel
from faststream.rabbit.fastapi import RabbitRouter, Logger  # type: ignore
from utils.credentials import load_rabbitmq_credentials
from tc_messageBroker.rabbit_mq.event import Event
from worker.tasks import query_data_sources

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds["url"])


class Incoming(BaseModel):
    question: str
    community_id: str


def call():
    return True


@router.subscriber(Event.HIVEMIND.INTERACTION_CREATED)
@router.publisher(Event.DISCORD_BOT.INTERACTION_RESPONSE.EDIT)
async def ask(m: Incoming, logger: Logger, d=Depends(call)):
    logger.info(f"COMMUNITY_ID: {m.community_id} Received job")
    response = query_data_sources(community_id=m.community_id, query=m.question)
    # TODO: find out how the publishing to DISCORD_BOT queue works
    return response
