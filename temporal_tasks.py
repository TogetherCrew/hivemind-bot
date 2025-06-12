from datetime import timedelta
import logging

from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.schema import NodeWithScore, TextNode
from schema import QuestionModel, ResponseModel, RouteModel, RouteModelPayload
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from utils.persist_payload import PersistPayload
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from worker.tasks import query_data_sources  # pylint: disable=no-name-in-module
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
from bot.evaluations.answer_relevance import AnswerRelevanceEvaluation
from bot.evaluations.answer_confidence import AnswerConfidenceEvaluation
from bot.evaluations.question_answered import QuestionAnswerCoverageEvaluation
from bot.evaluations.schema import AnswerRelevanceSuccess, AnswerConfidenceSuccess, QuestionAnswerCoverageSuccess


@activity.defn
async def run_hivemind_activity(payload: HivemindQueryPayload):
    response, references = query_data_sources(
        community_id=payload.community_id,
        query=payload.query,
        enable_answer_skipping=payload.enable_answer_skipping,
    )

    relevancy_result = await AnswerRelevanceEvaluation().evaluate(
        question=payload.query, answer=response
    )
    confidence_result = await AnswerConfidenceEvaluation().evaluate(
        question=payload.query, answer=response
    )
    coverage_result = await QuestionAnswerCoverageEvaluation().evaluate(
        question=payload.query, answer=response
    )
    response_payload = RouteModelPayload(
        communityId=payload.community_id,
        route=RouteModel(source="temporal", destination=None),
        question=QuestionModel(message=payload.query),
        response=ResponseModel(message=f"{response}\n\n{references}"),
        metadata={
            "answer_relevance_score": (
                relevancy_result.score
                if isinstance(relevancy_result, AnswerRelevanceSuccess)
                else relevancy_result.error
            ),
            "answer_relevance_explanation": (
                relevancy_result.explanation
                if isinstance(relevancy_result, AnswerRelevanceSuccess)
                else relevancy_result.error
            ),
            "answer_confidence_score": (
                confidence_result.score
                if isinstance(confidence_result, AnswerConfidenceSuccess)
                else confidence_result.error
            ),
            "answer_confidence_explanation": (
                confidence_result.explanation
                if isinstance(confidence_result, AnswerConfidenceSuccess)
                else confidence_result.error
            ),
            "answer_coverage_answered": (
                coverage_result.answered
                if isinstance(coverage_result, QuestionAnswerCoverageSuccess)
                else False
            ),
            "answer_coverage_score": (
                coverage_result.score
                if isinstance(coverage_result, QuestionAnswerCoverageSuccess)
                else coverage_result.error
            ),
            "answer_coverage_explanation": (
                coverage_result.explanation
                if isinstance(coverage_result, QuestionAnswerCoverageSuccess)
                else coverage_result.error
            ),
        },
    )

    # dumping the whole payload of question & answer to db
    persister = PersistPayload()
    persister.persist_payload(response_payload)

    # Hardcoded threshold for answer relevance
    # if the relevance score is less than 3, we do not return the answer
    # and in case of enable_answer_skipping is True (auto-answering questions)
    if (
        isinstance(coverage_result, QuestionAnswerCoverageSuccess)
        and coverage_result.score < 3
        and payload.enable_answer_skipping
    ):
        logging.warning(
            f"Answer coverage score is less than 3, skipping answer: {coverage_result.score}"
        )

        return None, []

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
