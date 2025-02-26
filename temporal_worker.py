import asyncio
import logging
import os

from dotenv import load_dotenv
from tc_temporal_backend.client import TemporalClient
from temporal_tasks import HivemindWorkflow, run_hivemind_activity
from temporalio.worker import UnsandboxedWorkflowRunner, Worker


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

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
        workflow_runner=UnsandboxedWorkflowRunner(),
    )

    logging.info("Starting worker...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
