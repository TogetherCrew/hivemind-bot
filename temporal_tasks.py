from datetime import timedelta

from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.schema import NodeWithScore, TextNode
from temporalio import activity, workflow
from openai import OpenAI
from temporalio.common import RetryPolicy
from utils.globals import NO_ANSWER_REFERENCE
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
from bot.agent.tools import rag_tool, general_llm_tool
import logging



async def hivemind_activity(payload: HivemindQueryPayload):
    """
    Route the incoming query to the appropriate tool.
    Uses a lightweight LLM-based router to select between RAG and general LLM.
    """
    try:
        client = OpenAI()
        router_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict router. Decide which tool to use for answering a question. "
                    "Return exactly one word: 'rag' if the question likely needs specific data, "
                    "or 'llm' if general world knowledge suffices. No extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {payload.query}\n\n"
                    "Choose tool:"
                ),
            },
        ]
        decision = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=router_messages,
            temperature=0.0,
        )
        choice = (decision.choices[0].message.content or "rag").strip().lower()
    except Exception as ex:
        logging.exception(f"Error routing question to tool. defaulting to rag. Exception: {ex}")
        choice = "rag"

    if choice == "llm":
        return await general_llm_tool(payload)
    else:
        return await rag_tool(payload)


@activity.defn
async def hivemind_temporal_activity(payload: HivemindQueryPayload):
    """
    wrapping the hivemind job as a temporal activity
    """
    return await hivemind_activity(payload)


@workflow.defn
class HivemindWorkflow:
    """
    a temporal workflow to run the hivemind querying data sources
    """

    @workflow.run
    async def run(self, payload: HivemindQueryPayload):
        response_tuple = await workflow.execute_activity(
            hivemind_temporal_activity,
            payload,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                maximum_interval=timedelta(minutes=5),
                maximum_attempts=1,
            ),
        )
        response, references = response_tuple

        references_nodes = self.serialize_references(references=references)
        answer_reference = ""
        if references and response != NO_ANSWER_REFERENCE:
            answer_reference = PrepareAnswerSources().prepare_answer_sources(
                nodes=references_nodes  # type: ignore
            )
        if response:
            return f"{response}\n\n{answer_reference}"
        else:
            return None

    def serialize_references(
        self, references: list[dict]
    ) -> list[SubQuestionAnswerPair]:
        ref_nodes: list[SubQuestionAnswerPair] = []
        for ref in references:
            answer = ref["answer"]
            sources = ref["sources"]
            sub_q = ref["sub_q"]

            sources_node = [
                NodeWithScore(node=TextNode(**src["node"]), score=src["score"])
                for src in sources
            ]

            ref_nodes.append(
                SubQuestionAnswerPair(sub_q=sub_q, answer=answer, sources=sources_node)
            )

        return ref_nodes
