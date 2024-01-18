import json
import logging
from typing import Any

from celery_app.server import app
from celery_app.utils.fire_event import job_send
from subquery import query_multiple_source
from tc_messageBroker.rabbit_mq.event import Event
from tc_messageBroker.rabbit_mq.payload.discord_bot.base_types.interaction_callback_data import (
    InteractionCallbackData,
)
from tc_messageBroker.rabbit_mq.payload.discord_bot.chat_input_interaction import (
    ChatInputCommandInteraction,
)
from tc_messageBroker.rabbit_mq.payload.discord_bot.interaction_response import (
    InteractionResponse,
)
from tc_messageBroker.rabbit_mq.payload.payload import Payload
from tc_messageBroker.rabbit_mq.queue import Queue


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
    create_interaction_content = Payload.DISCORD_BOT.INTERACTION_RESPONSE.Create(
        interaction=bot_given_info,
        data=InteractionResponse(
            type=4,
            data=InteractionCallbackData(
                content="Processing your question ...", flags=64
            ),
        ),
    ).to_dict()

    logging.info(f"{prefix}Sending process question to discord-bot!")
    job_send(
        event=Event.DISCORD_BOT.INTERACTION_RESPONSE.CREATE,
        queue_name=Queue.DISCORD_BOT,
        content=create_interaction_content,
    )
    logging.info(f"{prefix}Querying the data sources!")
    # for now we have just the discord platform
    response, source_nodes = query_multiple_source(
        query=question,
        community_id=community_id,
        discord=True,
    )

    source_nodes_dict: list[dict[str, Any]] = []
    for node in source_nodes:
        node_dict = dict(node)
        node_dict.pop("relationships", None)
        source_nodes_dict.append(node_dict)

    results = {
        "response": response,
        "source_nodes": source_nodes_dict,
    }

    interaction = json.loads(bot_given_info["content"]["interaction"])
    chat_input_interaction = ChatInputCommandInteraction.from_dict(interaction)

    response_payload = Payload.DISCORD_BOT.INTERACTION_RESPONSE.Edit(
        interaction=chat_input_interaction,
        data=InteractionCallbackData(content=json.dumps(results)),
    ).to_dict()

    logging.info(f"{prefix}Sending Edit response to discord-bot!")
    job_send(
        event=Event.DISCORD_BOT.INTERACTION_RESPONSE.EDIT,
        queue_name=Queue.DISCORD_BOT,
        content=response_payload,
    )
