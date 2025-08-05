from datetime import timedelta
import logging

from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.schema import NodeWithScore, TextNode
from schema import QuestionModel, ResponseModel, RouteModel, RouteModelPayload
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from utils.globals import NO_ANSWER_REFERENCE
from utils.persist_payload import PersistPayload
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources
from worker.tasks import query_data_sources  # pylint: disable=no-name-in-module
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
from bot.evaluations.answer_relevance import AnswerRelevanceEvaluation
from bot.evaluations.answer_confidence import AnswerConfidenceEvaluation
from bot.evaluations.question_answered import QuestionAnswerCoverageEvaluation
from bot.evaluations.node_relevance import NodeRelevanceEvaluation
from bot.evaluations.schema import (
    AnswerRelevanceSuccess,
    AnswerConfidenceSuccess,
    QuestionAnswerCoverageSuccess,
    AnswerRelevanceError,
    AnswerConfidenceError,
    QuestionAnswerCoverageError,
    NodeRelevanceSuccess,
)


@activity.defn
async def run_hivemind_activity(payload: HivemindQueryPayload):
    result = query_data_sources(
        community_id=payload.community_id,
        query=payload.query,
        enable_answer_skipping=payload.enable_answer_skipping,
        return_metadata=True,
    )

    response, references, metadata = result

    # Initialize node relevance evaluator
    node_evaluator = NodeRelevanceEvaluation()

    # Extract summary nodes and raw nodes from metadata
    summary_nodes: list[NodeWithScore] = []
    raw_nodes: list[NodeWithScore] = []

    for i in range(len(references)):
        # references is a list of SubQuestionAnswerPair
        raw_nodes.extend([node for node in references[i].sources])

    # Extract summary nodes from metadata if available
    # metadata structure: {'Telegram': {'summary_nodes': [node_list]}, 'Discord': {'summary_nodes': [node_list]}, ...}
    if metadata:
        for platform_name, platform_metadata in metadata.items():
            platform_summary_nodes = platform_metadata["summary_nodes"]
            if isinstance(platform_summary_nodes, list):
                # Filter out None values
                valid_summary_nodes = [
                    node for node in platform_summary_nodes if node is not None
                ]
                summary_nodes.extend(valid_summary_nodes)

    # Evaluate nodes
    summary_node_evaluations = []
    raw_node_evaluations = []
    nodes_evaluation_summary = None

    if summary_nodes:
        summary_node_evaluations = await node_evaluator.evaluate_nodes_batch(
            question=payload.query, nodes=summary_nodes, node_type="summary"
        )

    if raw_nodes:
        raw_node_evaluations = await node_evaluator.evaluate_nodes_batch(
            question=payload.query, nodes=raw_nodes, node_type="raw"
        )

    # Create nodes evaluation summary
    if summary_node_evaluations or raw_node_evaluations:
        nodes_evaluation_summary = node_evaluator.create_evaluation_summary(
            question=payload.query,
            summary_results=summary_node_evaluations,
            raw_results=raw_node_evaluations,
        )

    if response:
        relevancy_result = await AnswerRelevanceEvaluation().evaluate(
            question=payload.query, answer=response
        )
        confidence_result = await AnswerConfidenceEvaluation().evaluate(
            question=payload.query, answer=response
        )
        coverage_result = await QuestionAnswerCoverageEvaluation().evaluate(
            question=payload.query, answer=response
        )
    else:
        relevancy_result = AnswerRelevanceError(
            error="No response from the query engine",
            question=payload.query,
            answer=None,
        )
        confidence_result = AnswerConfidenceError(
            error="No response from the query engine",
            question=payload.query,
            answer=None,
        )
        coverage_result = QuestionAnswerCoverageError(
            error="No response from the query engine",
            question=payload.query,
            answer=None,
        )

    # Build metadata dictionary with all evaluations
    evaluation_metadata = {
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
    }

    # Add node evaluations to metadata
    if nodes_evaluation_summary:
        evaluation_metadata.update(
            {
                "nodes_total_count": nodes_evaluation_summary.total_nodes,
                "nodes_summary_count": nodes_evaluation_summary.summary_nodes_count,
                "nodes_raw_count": nodes_evaluation_summary.raw_nodes_count,
                "nodes_average_relevance_score": nodes_evaluation_summary.average_relevance_score,
                "nodes_high_relevance_count": nodes_evaluation_summary.high_relevance_nodes,
                "nodes_successful_evaluations": nodes_evaluation_summary.successful_evaluations,
                "nodes_failed_evaluations": nodes_evaluation_summary.failed_evaluations,
            }
        )

    # Add individual node evaluation results for detailed analysis
    if summary_node_evaluations:
        evaluation_metadata["summary_node_evaluations"] = [
            {
                "relevance_score": (
                    eval_result.relevance_score
                    if isinstance(eval_result, NodeRelevanceSuccess)
                    else None
                ),
                "explanation": (
                    eval_result.explanation
                    if isinstance(eval_result, NodeRelevanceSuccess)
                    else eval_result.error
                ),
                "node_id": (
                    eval_result.node_id
                    if hasattr(eval_result, "node_id")
                    else "unknown"
                ),
                "node_score": (
                    eval_result.node_score
                    if hasattr(eval_result, "node_score")
                    else 0.0
                ),
                "success": isinstance(eval_result, NodeRelevanceSuccess),
            }
            for eval_result in summary_node_evaluations
        ]

    if raw_node_evaluations:
        evaluation_metadata["raw_node_evaluations"] = [
            {
                "relevance_score": (
                    eval_result.relevance_score
                    if isinstance(eval_result, NodeRelevanceSuccess)
                    else None
                ),
                "explanation": (
                    eval_result.explanation
                    if isinstance(eval_result, NodeRelevanceSuccess)
                    else eval_result.error
                ),
                "node_id": (
                    eval_result.node_id
                    if hasattr(eval_result, "node_id")
                    else "unknown"
                ),
                "node_score": (
                    eval_result.node_score
                    if hasattr(eval_result, "node_score")
                    else 0.0
                ),
                "success": isinstance(eval_result, NodeRelevanceSuccess),
            }
            for eval_result in raw_node_evaluations
        ]


    # Prepare answer references for response
    answer_reference = ""
    if references and response != NO_ANSWER_REFERENCE:
        answer_reference = PrepareAnswerSources().prepare_answer_sources(
            nodes=references  # type: ignore
        )


    response_payload = RouteModelPayload(
        communityId=payload.community_id,
        route=RouteModel(source="temporal", destination=None),
        question=QuestionModel(message=payload.query),
        response=ResponseModel(message=f"{response}\n\n{answer_reference}"),
        metadata=evaluation_metadata,
    )

    # Get workflow ID and update the payload in the database
    # If workflow_id is None, insert new data; else update existing document with evaluation results and response
    workflow_id = getattr(payload, "workflow_id", None)
    persister = PersistPayload()
    persister.persist_payload(response_payload, workflow_id=workflow_id)

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
