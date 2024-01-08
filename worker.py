from typing import Any
from celery_app.tasks import ask_question_auto_search
from tc_messageBroker import RabbitMQ
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue
from utils.credentials import load_rabbitmq_credentials


def query_llm(recieved_data: dict[str, Any]):
    """
    # TODO: Find the question and community_id
    """
    ask_question_auto_search.delay(
        question="TODO: QUESTION",
        community_id="TODO",
        bot_given_info=recieved_data,
    )


def job_recieve(broker_url, port, username, password):
    rabbit_mq = RabbitMQ(
        broker_url=broker_url, port=port, username=username, password=password
    )

    # TODO: Update according to our requirements
    rabbit_mq.on_event(Event.HIVEMIND.INTERACTION_CREATED, query_llm)
    rabbit_mq.connect(Queue.HIVEMIND)
    rabbit_mq.consume(Queue.HIVEMIND)

    if rabbit_mq.channel is not None:
        rabbit_mq.channel.start_consuming()
    else:
        print("Connection to broker was not successful!")


if __name__ == "__main__":
    rabbit_creds = load_rabbitmq_credentials()
    username = rabbit_creds["user"]
    password = rabbit_creds["password"]
    broker_url = rabbit_creds["host"]
    port = rabbit_creds["port"]

    job_recieve(broker_url, port, username, password)
