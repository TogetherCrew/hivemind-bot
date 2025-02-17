import gc
import logging

from celery.signals import task_postrun, task_prerun
from llama_index.core.schema import NodeWithScore
from subquery import query_multiple_source
from utils.data_source_selector import DataSourceSelector
from utils.globals import NO_DATA_SOURCE_SELECTED, QUERY_ERROR_MESSAGE
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
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
        answer_sources = PrepareAnswerSources().prepare_answer_sources(nodes=references)
    except Exception:
        response = QUERY_ERROR_MESSAGE
        answer_sources = None
        logging.error(
            f"Errors raised while processing the question for community: {community_id}!"
        )

    return {
        "community_id": community_id,
        "question": query,
        "response": response,
        "references": answer_sources,
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
    enable_answer_skipping: bool = False,
) -> tuple[str | None, list[NodeWithScore | None]]:
    """
    ask questions with auto select platforms

    Parameters
    -------------
    community_id : str
        the community id data to use for answering
    query : str
        the user query to ask llm
    enable_answer_skipping : bool
        skip answering questions with non-relevant retrieved information
        having this, it could provide `None` for response and source_nodes

    Returns
    ---------
    response : str | None
        the LLM's response
        would be `None` in case there was no relevant information was available and enable_answer_skipping was True
    references : list[NodeWithScore]
        the references that the answers were coming from
        would be a list of `None`
        in case there was no relevant information was available and enable_answer_skipping was True
    """
    prefix = f"COMMUNITY_ID: {community_id}"
    logging.info(f"{prefix} Finding data sources to query to!")
    logging.info(
        f"{prefix} Answer skipping in case of non-relevant information: {enable_answer_skipping}"
    )
    selector = DataSourceSelector()
    data_sources = selector.select_data_source(community_id)

    references: list = []
    if data_sources or enable_answer_skipping:
        logging.info(f"Quering data sources: {list(data_sources.keys())}!")
        response, references = query_multiple_source(
            query=query,
            community_id=community_id,
            enable_answer_skipping=enable_answer_skipping,
            **data_sources,
        )
    else:
        logging.info(f"No data source selected!")
        response = NO_DATA_SOURCE_SELECTED
        references = []

    if enable_answer_skipping and not references:
        response = None

    return response, references
