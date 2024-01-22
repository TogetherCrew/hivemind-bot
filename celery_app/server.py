from celery import Celery
import os
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from utils.credentials import load_rabbitmq_credentials

rabbit_creds = load_rabbitmq_credentials()
user = rabbit_creds["user"]
password = rabbit_creds["password"]
host = rabbit_creds["host"]
port = rabbit_creds["port"]

load_dotenv()
otel_endpoint = os.getenv("TRACELOOP_BASE_URL")
Traceloop.init(api_endpoint=otel_endpoint, app_name="hivemind-server")

app = Celery("celery_app/tasks", broker=f"pyamqp://{user}:{password}@{host}:{port}//")
app.autodiscover_tasks(["celery_app"])
