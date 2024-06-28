from dotenv import load_dotenv
from traceloop.sdk import Traceloop
import os
import logging

load_dotenv()


def init_tracing():
    logging.info("Initializing traceloop...")
    print("Initializing traceloop...")
    otel_endpoint = os.getenv("TRACELOOP_BASE_URL")
    Traceloop.init(app_name="hivemind-server", api_endpoint=otel_endpoint)
    logging.info("Traceloop initialized.")
    print("Traceloop initialized.")
