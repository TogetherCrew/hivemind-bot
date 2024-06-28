import logging
from celery import Celery
from utils.credentials import load_rabbitmq_credentials

from dotenv import load_dotenv
from traceloop.sdk import Traceloop
import os

load_dotenv()

rabbit_creds = load_rabbitmq_credentials()
user = rabbit_creds["user"]
password = rabbit_creds["password"]
host = rabbit_creds["host"]
port = rabbit_creds["port"]


def init_tracing():
    logging.info("Initializing trace...")
    otel_endpoint = os.getenv("TRACELOOP_BASE_URL")
    Traceloop.init(app_name="hivemind-server", api_endpoint=otel_endpoint)
    logging.info("Trace initialized...")


app = Celery(
    "tasks",
    broker=f"pyamqp://{user}:{password}@{host}:{port}//",
    include=["worker.tasks"],
)

if __name__ == "__main__":
    app.start()
