from typing import Union, List
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import SubQuestionAnswerPair
from .base_evaluations import BaseEvaluation
from .schema import NodeRelevanceSuccess, NodeRelevanceError, NodesEvaluationSummary


class NodeRelevanceEvaluation(BaseEvaluation):
    def __init__(
        self, model: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.0
    ):
        """
        Initialize the node relevance evaluation class.

        Parameters
        ----------
        model : str, optional
            The OpenAI model to use for evaluations, by default "gpt-4.1-mini-2025-04-14"
        temperature : float, optional
            Temperature for model responses, by default 0.0
        """
        super().__init__(model, temperature)

    async def evaluate_nodes_batch_together(
        self, question: str, nodes: List[NodeWithScore], node_type: str = "unknown"
    ) -> List[Union[NodeRelevanceSuccess, NodeRelevanceError]]:
        """
        Evaluate the relevance of multiple nodes to a given question in a single API call.

        Parameters
        ----------
        question : str
            The question that was asked
        nodes : List[NodeWithScore]
            The nodes to evaluate
        node_type : str, optional
            Type of nodes being evaluated (e.g., "summary", "raw"), by default "unknown"

        Returns
        -------
        List[Union[NodeRelevanceSuccess, NodeRelevanceError]]
            List of evaluation results for each node
        """
        if not nodes:
            return []

        system_message = self._create_system_message("node relevance evaluation")

        # Prepare all nodes for batch evaluation
        nodes_info = []
        for i, node in enumerate(nodes):
            node_content = node.node.get_content()
            node_id = getattr(node.node, "node_id", f"node_{i}")
            node_score = node.score or 0.0

            nodes_info.append(
                {
                    "index": i,
                    "content": node_content,
                    "id": node_id,
                    "score": node_score,
                }
            )

        # Create batch evaluation prompt
        evaluation_prompt = f"""Please evaluate how relevant each of the following retrieved texts is to the question on a scale of 1-10.

Consider for each text:
1. Does the text contain information that directly answers or relates to the question?
2. Is the information in the text useful for answering the question?
3. How specific and detailed is the relevant information?
4. Does the text provide context that would help understand the answer?
5. Is there any useful background information related to the question?

Question: {question}

Retrieved Texts ({node_type} nodes):
"""

        for info in nodes_info:
            evaluation_prompt += f"""

Node {info['index'] + 1} (ID: {info['id']}, Similarity Score: {info['score']:.3f}):
{info['content'][:500]}{'...' if len(info['content']) > 500 else ''}
"""

        evaluation_prompt += f"""

Provide your evaluation for ALL {len(nodes)} nodes in the following format:
Node 1:
Score: [1-10]
Explanation: [Brief explanation]

Node 2:
Score: [1-10]
Explanation: [Brief explanation]

[Continue for all nodes...]
"""

        messages = [system_message, {"role": "user", "content": evaluation_prompt}]

        try:
            print(f"Evaluating {len(nodes)} {node_type} nodes together...")
            response = await self._get_llm_response(messages)

            # Parse the batch response
            results = []
            response_lines = response.split("\n")
            current_node_idx = 0

            for i, line in enumerate(response_lines):
                if line.strip().startswith(f"Node {current_node_idx + 1}:"):
                    # Look for Score and Explanation in subsequent lines
                    score_line = None
                    explanation_line = None

                    for j in range(i + 1, min(i + 10, len(response_lines))):
                        if response_lines[j].strip().startswith("Score:"):
                            score_line = response_lines[j]
                        elif response_lines[j].strip().startswith("Explanation:"):
                            explanation_line = response_lines[j]
                            break

                    if (
                        score_line
                        and explanation_line
                        and current_node_idx < len(nodes_info)
                    ):
                        try:
                            score = int(score_line.split(":")[1].strip())
                            explanation = explanation_line.split(":", 1)[1].strip()

                            node_info = nodes_info[current_node_idx]
                            results.append(
                                NodeRelevanceSuccess(
                                    relevance_score=score,
                                    explanation=explanation,
                                    question=question,
                                    node_content=node_info["content"],
                                    node_id=node_info["id"],
                                    node_score=node_info["score"],
                                )
                            )
                        except (ValueError, IndexError) as e:
                            node_info = nodes_info[current_node_idx]
                            results.append(
                                NodeRelevanceError(
                                    error=f"Failed to parse node {current_node_idx + 1}: {str(e)}",
                                    raw_response=f"Score: {score_line}, Explanation: {explanation_line}",
                                    question=question,
                                    node_content=node_info["content"],
                                    node_id=node_info["id"],
                                    node_score=node_info["score"],
                                )
                            )
                    else:
                        # Failed to parse this node
                        if current_node_idx < len(nodes_info):
                            node_info = nodes_info[current_node_idx]
                            results.append(
                                NodeRelevanceError(
                                    error=f"Failed to find score/explanation for node {current_node_idx + 1}",
                                    raw_response=response,
                                    question=question,
                                    node_content=node_info["content"],
                                    node_id=node_info["id"],
                                    node_score=node_info["score"],
                                )
                            )

                    current_node_idx += 1

            # If we didn't parse all nodes, add errors for the remaining ones
            while len(results) < len(nodes_info):
                node_info = nodes_info[len(results)]
                results.append(
                    NodeRelevanceError(
                        error=f"No evaluation found for node {len(results) + 1}",
                        raw_response=response,
                        question=question,
                        node_content=node_info["content"],
                        node_id=node_info["id"],
                        node_score=node_info["score"],
                    )
                )

            return results

        except Exception as e:
            # If batch evaluation fails, return errors for all nodes
            results = []
            for info in nodes_info:
                results.append(
                    NodeRelevanceError(
                        error=f"Batch evaluation failed: {str(e)}",
                        raw_response=None,
                        question=question,
                        node_content=info["content"],
                        node_id=info["id"],
                        node_score=info["score"],
                    )
                )
            return results

    async def evaluate_nodes_batch(
        self, question: str, nodes: List[NodeWithScore], node_type: str = "unknown"
    ) -> List[Union[NodeRelevanceSuccess, NodeRelevanceError]]:
        """
        Evaluate the relevance of multiple nodes to a given question.

        Parameters
        ----------
        question : str
            The question that was asked
        nodes : List[NodeWithScore]
            The nodes to evaluate
        node_type : str, optional
            Type of nodes being evaluated (e.g., "summary", "raw"), by default "unknown"

        Returns
        -------
        List[Union[NodeRelevanceSuccess, NodeRelevanceError]]
            List of evaluation results for each node
        """
        # Convert all nodes to NodeWithScore format
        node_with_scores = []

        for node in nodes:
            node_with_scores.append(node)

        # Filter out None values
        node_with_scores = [n for n in node_with_scores if n is not None]

        # Use the batch evaluation method
        return await self.evaluate_nodes_batch_together(
            question, node_with_scores, node_type
        )

    async def evaluate(self, **kwargs):
        """
        Main evaluate method for compatibility with base class.
        Use evaluate_nodes_batch for specific functionality.
        """
        if "nodes" in kwargs and "question" in kwargs:
            return await self.evaluate_nodes_batch(
                kwargs["question"], kwargs["nodes"], kwargs.get("node_type", "unknown")
            )
        else:
            raise ValueError("'nodes' and 'question' must be provided")

    def create_evaluation_summary(
        self,
        question: str,
        summary_results: List[Union[NodeRelevanceSuccess, NodeRelevanceError]],
        raw_results: List[Union[NodeRelevanceSuccess, NodeRelevanceError]],
    ) -> NodesEvaluationSummary:
        """
        Create a summary of node evaluations.

        Parameters
        ----------
        question : str
            The question that was asked
        summary_results : List[Union[NodeRelevanceSuccess, NodeRelevanceError]]
            Results from evaluating summary nodes
        raw_results : List[Union[NodeRelevanceSuccess, NodeRelevanceError]]
            Results from evaluating raw nodes

        Returns
        -------
        NodesEvaluationSummary
            Summary statistics of the node evaluations
        """
        all_results = summary_results + raw_results

        successful_results = [
            r for r in all_results if isinstance(r, NodeRelevanceSuccess)
        ]
        failed_results = [r for r in all_results if isinstance(r, NodeRelevanceError)]

        total_nodes = len(all_results)
        summary_nodes_count = len(summary_results)
        raw_nodes_count = len(raw_results)
        successful_evaluations = len(successful_results)
        failed_evaluations = len(failed_results)

        # Calculate average relevance score for successful evaluations
        if successful_results:
            average_relevance_score = sum(
                r.relevance_score for r in successful_results
            ) / len(successful_results)
            high_relevance_nodes = sum(
                1 for r in successful_results if r.relevance_score >= 7
            )
        else:
            average_relevance_score = 0.0
            high_relevance_nodes = 0

        return NodesEvaluationSummary(
            question=question,
            total_nodes=total_nodes,
            summary_nodes_count=summary_nodes_count,
            raw_nodes_count=raw_nodes_count,
            average_relevance_score=average_relevance_score,
            high_relevance_nodes=high_relevance_nodes,
            successful_evaluations=successful_evaluations,
            failed_evaluations=failed_evaluations,
        )
