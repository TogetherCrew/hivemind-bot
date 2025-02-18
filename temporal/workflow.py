from pydantic import BaseModel, Field
from temporalio import activity, workflow
from worker.tasks import query_data_sources


class HivemindQueryPayload(BaseModel):
    community_id: str = Field(..., "the community id data to use for answering")
    query: str = Field(..., "the user query to ask llm")
    enable_answer_skipping: bool = Field(
        False,
        (
            "skip answering questions with non-relevant retrieved information"
            "having this, it could provide `None` for response and source_nodes"
        ),
    )


@activity.defn
def run_hivemind_activity(payload: HivemindQueryPayload):
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
        response_tuple = await workflow.execute_activity(run_hivemind_activity, payload)
        return response_tuple
