from fastapi import Depends
from pydantic import BaseModel
from faststream.rabbit.fastapi import RabbitRouter, Logger # type: ignore
from utils.credentials import load_rabbitmq_credentials
from tc_messageBroker.rabbit_mq.event import Event

rabbitmq_creds = load_rabbitmq_credentials()

router = RabbitRouter(rabbitmq_creds['url'])

class Incoming(BaseModel):
    m: dict

def call():
    return True

@router.subscriber(Event.HIVEMIND.INTERACTION_CREATED)
@router.publisher(Event.DISCORD_BOT.INTERACTION_RESPONSE.EDIT)
async def ask(m: Incoming, logger: Logger, d=Depends(call)):
    logger.info(m)
    return { "response": "Hello, Rabbit!" }