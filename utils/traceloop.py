import logging
import os

from dotenv import load_dotenv
from traceloop.sdk import Traceloop

load_dotenv()


def init_tracing():
    otel_endpoint = os.getenv("TRACELOOP_BASE_URL")
    if not otel_endpoint:
        logging.error("TRACELOOP_BASE_URL is not set.")
        return
    Traceloop.init(app_name="hivemind-worker", api_endpoint=otel_endpoint)
    logging.info("Traceloop initialized.")
