from tc_messageBroker import RabbitMQ
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue

if __name__ == "__main__":
    broker_url = "localhost"
    port = 5672
    username = "root"
    password = "pass"

    rabbit_mq = RabbitMQ(
        broker_url=broker_url, port=port, username=username, password=password
    )

    rabbit_mq.connect(Queue.HIVEMIND, queue_durable=False)

    content = {
        "community_id": "****",
        "question": "What is Hivemind?",
    }

    rabbit_mq.publish(
        Queue.HIVEMIND,
        event=Event.DISCORD_ANALYZER.RUN,
        content=content,
    )
