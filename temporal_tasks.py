from datetime import timedelta

from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.schema import NodeWithScore, TextNode
from pydantic import BaseModel, Field
from schema import QuestionModel, ResponseModel, RouteModel, RouteModelPayload
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from utils.persist_payload import PersistPayload
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from worker.tasks import query_data_sources  # pylint: disable=no-name-in-module


class HivemindQueryPayload(BaseModel):
    community_id: str = Field(
        ..., description="the community id data to use for answering"
    )
    query: str = Field(..., description="the user query to ask llm")
    enable_answer_skipping: bool = Field(
        False,
        description=(
            "skip answering questions with non-relevant retrieved information"
            "having this, it could provide `None` for response and source_nodes"
        ),
    )


@activity.defn
async def run_hivemind_activity(payload: HivemindQueryPayload):
    response, references = query_data_sources(
        community_id=payload.community_id,
        query=payload.query,
        enable_answer_skipping=payload.enable_answer_skipping,
    )

    response_payload = RouteModelPayload(
        communityId=payload.community_id,
        route=RouteModel(source="temporal", destination=None),
        question=QuestionModel(message=payload.query),
        response=ResponseModel(message=f"{response}\n\n{references}"),
        metadata=None,
    )

    # dumping the whole payload of question & answer to db
    persister = PersistPayload()
    persister.persist_payload(response_payload)

    return response, references


@workflow.defn
class HivemindWorkflow:
    """
    a temporal workflow to run the hivemind querying data sources
    """

    @workflow.run
    async def run(self, payload: HivemindQueryPayload):
        response_tuple = await workflow.execute_activity(
            run_hivemind_activity,
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
        answer_reference = PrepareAnswerSources().prepare_answer_sources(
            nodes=references_nodes
        )

        return f"{response}\n\n{answer_reference}"

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
