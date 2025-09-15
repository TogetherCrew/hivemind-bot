import gc
import logging
from typing import Any

from celery.signals import task_postrun, task_prerun
from llama_index.core.query_engine import SubQuestionAnswerPair
from subquery import query_multiple_source
from utils.data_source_selector import DataSourceSelector
from utils.globals import (
    NO_ANSWER_REFERENCE,
    NO_DATA_SOURCE_SELECTED,
    QUERY_ERROR_MESSAGE,
)
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from utils.traceloop import init_tracing
from worker.celery import app
from utils.rephrase import rephrase_question


@app.task
def ask_question_auto_search(
    community_id: str,
    query: str,
) -> dict[str, Any]:
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
    return_metadata: bool = False,
) -> (
    tuple[str | None, list[SubQuestionAnswerPair | None]]
    | tuple[str | None, list[SubQuestionAnswerPair | None], dict]
):
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
    return_metadata : bool
        return metadata from the query engines

    Returns
    ---------
    response : str | None
        the LLM's response
        would be `None` in case there was no relevant information was available and enable_answer_skipping was True
    references : list[Sub]
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
    logging.info(f"{prefix} Data sources selected: {data_sources}")

    # Platform IDs are now directly in data_sources, pass them directly
    # No need to convert to boolean values

    references: list = []
    metadata = {}
    if data_sources or enable_answer_skipping:
        logging.info(f"Quering data sources: {list(data_sources.keys())}!")
        result = query_multiple_source(
            query=query,
            community_id=community_id,
            enable_answer_skipping=enable_answer_skipping,
            return_metadata=return_metadata,
            **data_sources,
        )
        if return_metadata:
            response, references, metadata = result
        else:
            response, references = result

        # If no useful answer, try rephrasing once and retry
        no_answer = (
            response is None
            or response == NO_ANSWER_REFERENCE
            or not references
        )
        if no_answer:
            rephrased = rephrase_question(query, context_hint=f"community_id={community_id}")
            if rephrased and rephrased != query:
                logging.info(f"{prefix} Rephrasing query and retrying: '{rephrased}'")
                result = query_multiple_source(
                    query=rephrased,
                    community_id=community_id,
                    enable_answer_skipping=enable_answer_skipping,
                    return_metadata=return_metadata,
                    **data_sources,
                )
                if return_metadata:
                    response, references, metadata = result
                else:
                    response, references = result
    else:
        logging.info(f"No data source selected!")
        response = NO_DATA_SOURCE_SELECTED
        references = []

    if enable_answer_skipping and (not references or response == NO_ANSWER_REFERENCE):
        response = None

    if return_metadata:
        return response, references, metadata

    return response, references
