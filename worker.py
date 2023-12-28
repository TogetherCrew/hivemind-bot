from tc_messageBroker import RabbitMQ
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.queue import Queue

from celery_app.tasks import add


# TODO: Update according to our requirements
def do_something(recieved_data):
    message = f"Calculation Results:"
    print(message)
    print(f"recieved_data: {recieved_data}")
    add.delay(20, 14)


def job_recieve(broker_url, port, username, password):
    rabbit_mq = RabbitMQ(
        broker_url=broker_url, port=port, username=username, password=password
    )

    # TODO: Update according to our requirements
    rabbit_mq.on_event(Event.HIVEMIND.INTERACTION_CREATED, do_something)
    rabbit_mq.connect(Queue.HIVEMIND)
    rabbit_mq.consume(Queue.HIVEMIND)

    if rabbit_mq.channel is not None:
        rabbit_mq.channel.start_consuming()
    else:
        print("Connection to broker was not successful!")


if __name__ == "__main__":
    # TODO: read from .env
    broker_url = "localhost"
    port = 5672
    username = "root"
    password = "pass"

    job_recieve(broker_url, port, username, password)
