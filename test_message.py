from tc_messageBroker.rabbit_mq.event import Event
import asyncio
import aio_pika
import json


async def main() -> None:
    connection = await aio_pika.connect_robust(
        "amqp://root:pass@127.0.0.1:5672/",
    )

    async with connection:
        routing_key = Event.HIVEMIND.INTERACTION_CREATED

        channel = await connection.channel()

        payload = {
            "m": { "hello": "world!" }
        }

        body = json.dumps(payload).encode("utf-8")

        await channel.default_exchange.publish(
            aio_pika.Message(body=body),
            routing_key=routing_key,
        )


if __name__ == "__main__":
    asyncio.run(main())