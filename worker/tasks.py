import gc
import logging

from celery.signals import task_postrun, task_prerun
from subquery import query_multiple_source
from llama_index.core.schema import NodeWithScore
from utils.data_source_selector import DataSourceSelector
from utils.traceloop import init_tracing
from worker.celery import app


@app.task
def ask_question_auto_search(
    community_id: str,
    query: str,
) -> dict[str, str]:
    try:
        response, references = query_data_sources(
            community_id=community_id, query=query
        )
    except Exception:
        response = "Sorry, We cannot process your question at the moment."
        logging.error(
            f"Errors raised while processing the question for community: {community_id}!"
        )

    return {
        "community_id": community_id,
        "question": query,
        "response": response,
        "references": references,
    }


@task_prerun.connect
def task_prerun_handler(sender=None, **kwargs):
    # Initialize Traceloop for LLM
    init_tracing()


@task_postrun.connect
def task_postrun_handler(sender=None, **kwargs):
    # Trigger garbage collection after each task
    gc.collect()


def query_data_sources(
    community_id: str,
    query: str,
) -> tuple[str, list[NodeWithScore]]:
    """
    ask questions with auto select platforms

    Parameters
    -------------
    community_id : str
        the community id data to use for answering
    query : str
        the user query to ask llm

    Returns
    ---------
    response : str
        the LLM's response
    references : list[NodeWithScore]
        the references that the answers were coming from
    """
    logging.info(f"COMMUNITY_ID: {community_id} Finding data sources to query to!")
    selector = DataSourceSelector()
    data_sources = selector.select_data_source(community_id)
    logging.info(f"Quering data sources: {data_sources}!")
    response, references = query_multiple_source(
        query=query,
        community_id=community_id,
        **data_sources,
    )

    return response, references
