import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.worker import Worker
from temporal.temporal_client import TemporalClient
from temporal.workflow import run_hivemind_activity, HivemindWorkflow


async def main():
    # Initialize environment
    load_dotenv()
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE")
    if not task_queue:
        raise ValueError("`TEMPORAL_TASK_QUEUE` is not properly set!")

    logging.info(f"Using task queue: {task_queue}")
    client = await TemporalClient().get_client()

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[HivemindWorkflow],
        activities=[run_hivemind_activity],
    )

    logging.info("Starting worker...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
