from celery import Celery

# TODO: read from .env
app = Celery("celery_app/tasks", broker="pyamqp://root:pass@localhost//")
app.autodiscover_tasks(["celery_app"])
