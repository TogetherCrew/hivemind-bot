from celery_app.job_send import job_send
from celery_app.server import app

# TODO: Write tasks that match our requirements


@app.task
def add(x, y):
    broker_url = "localhost"
    port = 5672
    username = "root"
    password = "pass"

    res = x + y
    job_send(broker_url, port, username, password, res)

    return res


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)
