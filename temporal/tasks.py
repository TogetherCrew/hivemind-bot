from datetime import timedelta

from pydantic import BaseModel, Field
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
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
        return response_tuple
