from celery_app.job_send import job_send
from celery_app.server import app
from utils.credentials import load_rabbitmq_credentials

# TODO: Write tasks that match our requirements


@app.task
def add(x, y):
    rabbit_creds = load_rabbitmq_credentials()
    username = rabbit_creds["user"]
    password = rabbit_creds["password"]
    broker_url = rabbit_creds["host"]
    port = rabbit_creds["port"]

    res = x + y
    job_send(broker_url, port, username, password, res)

    return res


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)
