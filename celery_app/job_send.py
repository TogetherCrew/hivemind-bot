from tc_messageBroker import RabbitMQ
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue


def job_send(broker_url, port, username, password, res):
    rabbit_mq = RabbitMQ(
        broker_url=broker_url, port=port, username=username, password=password
    )

    content = {
        "uuid": "d99a1490-fba6-11ed-b9a9-0d29e7612dp8",
        "data": f"some results {res}",
    }

    rabbit_mq.connect(Queue.DISCORD_ANALYZER)
    rabbit_mq.publish(
        queue_name=Queue.DISCORD_ANALYZER,
        event=Event.DISCORD_BOT.FETCH,
        content=content,
    )


if __name__ == "__main__":
    # TODO: read from .env
    broker_url = "localhost"
    port = 5672
    username = "root"
    password = "pass"
    job_send(broker_url, port, username, password, "CALLED FROM __main__")
