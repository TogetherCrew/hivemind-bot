import json
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
from tc_messageBroker.rabbit_mq.payload.payload import Payload
from tc_messageBroker.rabbit_mq.queue import Queue


@app.task
def ask_question_auto_search(
    question: str,
    community_id: str,
    bot_given_info: ChatInputCommandInteraction,
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
        This would be the `ChatInputCommandInteraction`.
    """

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

    response_payload = Payload.DISCORD_BOT.INTERACTION_RESPONSE.Create(
        type=19,
        data=InteractionCallbackData(content=json.dumps(results)),
        interaction=bot_given_info.to_dict(),
    ).to_dict()

    job_send(
        event=Event.DISCORD_BOT.INTERACTION_RESPONSE.CREATE,
        queue_name=Queue.DISCORD_BOT,
        content=response_payload,
    )
