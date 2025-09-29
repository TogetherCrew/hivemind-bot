from datetime import timedelta

from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.schema import NodeWithScore, TextNode
from temporalio import activity, workflow
from openai import OpenAI
from temporalio.common import RetryPolicy
from utils.globals import NO_ANSWER_REFERENCE, NO_ANSWER_REFERENCE_PLACEHOLDER
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
from bot.agent.tools import rag_tool, general_llm_tool
import logging



async def hivemind_activity(payload: HivemindQueryPayload):
    """
    Route the incoming query to the appropriate tool.
    Uses a lightweight LLM-based router to select between RAG and general LLM.
    """
    # If answer skipping is enabled, always route to RAG directly
    if payload.enable_answer_skipping:
        return await rag_tool(payload)

    # else, try to answer using the general knowledge as well
    try:
        client = OpenAI()
        router_messages = [
            {
                "role": "system",
                "content": (
                    "You are the TogetherCrew bot which is a strict router and lightweight answerer. If the question requires specific, external, or context data "
                    "(e.g., company/product/project/community/user/platform/etc.), return exactly one word: 'rag'. "
                    "Do not act as a general-purpose assistant and do not answer very broad or generic questions; if the query is generic, out of scope, or not directly about helping the user's specific problem, return 'rag'. "
                    "Otherwise, answer the question directly in one short paragraph or less, following these rules: "
                    "rely solely on your own knowledge, do not fabricate citations or sources, provide concise and clear answers, "
                    f"never provide suggestions or ask for clarifications, and if you don't know the answer, reply exactly with '{NO_ANSWER_REFERENCE_PLACEHOLDER}'. "
                    "When in doubt, prefer returning 'rag'. Return only 'rag' or the answer text. No extra commentary."
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
        router_output = (decision.choices[0].message.content or "rag").strip()
    except Exception as ex:
        logging.exception(f"Error routing question to tool. defaulting to rag. Exception: {ex}")
        router_output = "rag"

    if router_output.lower() == "rag":
        return await rag_tool(payload)
    # For any non-'rag' output, we answer via general LLM tool to ensure
    # consistent evaluations and persistence behavior.
    return await general_llm_tool(payload)


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
