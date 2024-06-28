from celery import Celery
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
import os

from utils.credentials import load_rabbitmq_credentials

rabbit_creds = load_rabbitmq_credentials()
user = rabbit_creds["user"]
password = rabbit_creds["password"]
host = rabbit_creds["host"]
port = rabbit_creds["port"]

app = Celery(
    "tasks",
    broker=f"pyamqp://{user}:{password}@{host}:{port}//",
    include=["worker.tasks"],
)
# app.autodiscover_tasks(["celery_app"])

if __name__ == "__main__":
    load_dotenv()
    otel_endpoint = os.getenv("TRACELOOP_BASE_URL")
    Traceloop.init(app_name="hivemind-server", api_endpoint=otel_endpoint)
    app.start()
