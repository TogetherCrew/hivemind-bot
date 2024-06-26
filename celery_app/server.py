from celery import Celery
from utils.credentials import load_rabbitmq_credentials

rabbit_creds = load_rabbitmq_credentials()
user = rabbit_creds["user"]
password = rabbit_creds["password"]
host = rabbit_creds["host"]
port = rabbit_creds["port"]

app = Celery(
    "celery_app/tasks", broker=f"librabbitmq://{user}:{password}@{host}:{port}//"
)
app.autodiscover_tasks(["celery_app"])
