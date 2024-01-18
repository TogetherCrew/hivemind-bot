import logging
from typing import Any

from tc_messageBroker import RabbitMQ
from utils.credentials import load_rabbitmq_credentials


def job_send(event: str, queue_name: str, content: dict[str, Any]) -> None:
    """
    fire the data to a specific event on a specific queue

    Parameters
    -----------
    event : str
        the event to fire message to
    queue_name : str
        the queue to fire message on
    content : dict[str, Any]
        the content to send messages to
    """
    logging.info(f"IN JOB_SEND!, event: {event}")
    print(f"JOB_SEND: CONTENT: {content}")

    rabbit_creds = load_rabbitmq_credentials()
    username = rabbit_creds["user"]
    password = rabbit_creds["password"]
    broker_url = rabbit_creds["host"]
    port = rabbit_creds["port"]
    rabbit_mq = RabbitMQ(
        broker_url=broker_url, port=port, username=username, password=password
    )
    logging.info("Connecting to rabbitMQ!")
    rabbit_mq.connect(queue_name)
    logging.info("Trying to publish on rabbitMQ")
    rabbit_mq.publish(
        queue_name=queue_name,
        event=event,
        content=content,
    )
    logging.info("Published to RabbitMQ!")
