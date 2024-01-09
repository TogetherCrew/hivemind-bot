from typing import Any
from celery_app.tasks import ask_question_auto_search
from tc_messageBroker import RabbitMQ
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue
from utils.credentials import load_rabbitmq_credentials
from tc_messageBroker.rabbit_mq.payload.discord_bot.chat_input_interaction import (
    ChatInputCommandInteraction,
)
from utils.fetch_community_id import fetch_community_id_by_guild_id


def query_llm(recieved_data: dict[str, Any]):
    """
    query the llm using the received data
    """
    recieved_input = ChatInputCommandInteraction.from_dict(user_input)
    user_input = recieved_input.options["_hoistedOptions"]["value"]

    community_id = fetch_community_id_by_guild_id(guild_id=recieved_input.guild_id)
    ask_question_auto_search.delay(
        question=user_input,
        community_id=community_id,
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
