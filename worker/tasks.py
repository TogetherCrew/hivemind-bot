import gc
import json
import logging
from typing import Any

from celery.signals import task_postrun, worker_process_init
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.payload.discord_bot.base_types.interaction_callback_data import (
    InteractionCallbackData,
)
from tc_messageBroker.rabbit_mq.payload.discord_bot.chat_input_interaction import (
    ChatInputCommandInteraction,
)
from tc_messageBroker.rabbit_mq.payload.payload import Payload
from tc_messageBroker.rabbit_mq.queue import Queue

from subquery import query_multiple_source
from utils.data_source_selector import DataSourceSelector
from worker.utils.fire_event import job_send
from worker.celery import app

from dotenv import load_dotenv
from traceloop.sdk import Traceloop
import os


@worker_process_init.connect(weak=False)
def init_tracing():
    logging.info("Initializing trace...")
    load_dotenv()
    otel_endpoint = os.getenv("TRACELOOP_BASE_URL")
    Traceloop.init(app_name="hivemind-server", api_endpoint=otel_endpoint)
    logging.info("Trace initialized...")


@app.task
def ask_question_auto_search(
    question: str,
    community_id: str,
    bot_given_info: dict[str, Any],
) -> None:
    """
    this task is for the case that the user asks a question
    it would first retrieve the search metadata from summaries
    then perform a query on the filetred raw data to find answer

    Parameters
    ------------
    question : str
        the user question
    community_id : str
        the community that the question was asked in
    bot_given_info : tc_messageBroker.rabbit_mq.payload.discord_bot.chat_input_interaction.ChatInputCommandInteraction
        the information data that needed to be sent back to the bot again.
        This would be a dictionary representing the keys
        - `event`
        - `date`
        - `content`: which is the `ChatInputCommandInteraction` as a dictionary
    """

    prefix = f"COMMUNITY_ID: {community_id} | "
    logging.info(f"{prefix}Processing question!")
    interaction = json.loads(bot_given_info["content"]["interaction"])
    chat_input_interaction = ChatInputCommandInteraction.from_dict(interaction)

    try:
        # create_interaction_content = Payload.DISCORD_BOT.INTERACTION_RESPONSE.Create(
        #     interaction=chat_input_interaction,
        #     data=InteractionResponse(
        #         type=4,
        #         data=InteractionCallbackData(
        #             content="Processing your question ...", flags=64
        #         ),
        #     ),
        # ).to_dict()

        # logging.info(f"{prefix}Sending process question to discord-bot!")
        # job_send(
        #     event=Event.DISCORD_BOT.INTERACTION_RESPONSE.CREATE,
        #     queue_name=Queue.DISCORD_BOT,
        #     content=create_interaction_content,
        # )
        logging.info(f"{prefix}Querying the data sources!")
        # for now we have just the discord platform
        selector = DataSourceSelector()
        data_sources = selector.select_data_source(community_id)
        response, _ = query_multiple_source(
            query=question,
            community_id=community_id,
            **data_sources,
        )

        # source_nodes_dict: list[dict[str, Any]] = []
        # for node in source_nodes:
        #     node_dict = dict(node)
        #     node_dict.pop("relationships", None)
        #     source_nodes_dict.append(node_dict)

        # results = {
        # "response": response,
        # The source of answers is commented for now
        # "source_nodes": source_nodes_dict,
        # }
        results = f"**Question:** {question}\n**Answer:** {response}"

        response_payload = Payload.DISCORD_BOT.INTERACTION_RESPONSE.Edit(
            interaction=chat_input_interaction,
            data=InteractionCallbackData(
                # content=json.dumps(results)
                content=results
            ),
        ).to_dict()

        logging.info(f"{prefix}Sending Edit response to discord-bot!")
        job_send(
            event=Event.DISCORD_BOT.INTERACTION_RESPONSE.EDIT,
            queue_name=Queue.DISCORD_BOT,
            content=response_payload,
        )
        logging.info("FINISHED JOB")
    except Exception as exp:
        logging.error(f"Exception {exp} | during processing the question {question}")
        response_payload = Payload.DISCORD_BOT.INTERACTION_RESPONSE.Edit(
            interaction=chat_input_interaction,
            data=InteractionCallbackData(
                content="Sorry, We cannot process your question at the moment."
            ),
        ).to_dict()
        job_send(
            event=Event.DISCORD_BOT.INTERACTION_RESPONSE.EDIT,
            queue_name=Queue.DISCORD_BOT,
            content=response_payload,
        )
        logging.info("FINISHED JOB WITH EXCEPTION")


@task_postrun.connect
def task_postrun_handler(sender=None, **kwargs):
    # Trigger garbage collection after each task
    gc.collect()
