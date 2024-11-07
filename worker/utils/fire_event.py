from typing import Any
from faststream.rabbit import RabbitBroker

from utils.credentials import load_rabbitmq_credentials


async def job_send(message: Any, queue_name: str) -> None:
    """
    fire the data to a specific event on a specific queue

    Parameters
    -----------
    message : Any
        the message to be sent
    queue_name : str
        the queue to fire message on
    """
    rabbitmq_creds = load_rabbitmq_credentials()
    broker = RabbitBroker(url=rabbitmq_creds["url"])

    await broker.publish(message=message, queue=queue_name)
