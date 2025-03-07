import logging
from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.schema import NodeWithScore
from utils.globals import REFERENCE_SCORE_THRESHOLD


class PrepareAnswerSources:
    def __init__(
        self, threshold: float = REFERENCE_SCORE_THRESHOLD, max_references: int = 3
    ) -> None:
        """
        Initialize the PrepareAnswerSources class.

        Parameters
        ----------
        threshold : float, optional
            Minimum score threshold for including a node's URL, by default 0.5 set in globals file
        max_references : int, optional
            Maximum number of references to include, by default 3
        """
        self.threshold = threshold
        self.max_references = max_references

    def prepare_answer_sources(self, nodes: list[SubQuestionAnswerPair | None]) -> str:
        """
        Prepares a formatted string containing unique source URLs
        from the provided nodes, avoiding duplicate URLs.

        Parameters
        ----------
        nodes : list[SubQuestionAnswerPair]
            A list of node collections used for answering a question. Each node collection
            contains:
            - sub_q.tool_name: Name of the tool that generated these nodes
            - sources: List of nodes, each containing:
                - score: Relevance score of the node
                - metadata: Dictionary containing an optional 'url' field

        Returns
        -------
        all_sources : str
            A formatted string containing numbered URLs with the format:
            Top `x` references:
            [1] {url1}
            [2] {url2}

            Returns an empty string if:
            - The input nodes list is empty
            - No nodes meet the score threshold
            - No valid URLs are found in the nodes' metadata
        """
        # Return early if no nodes
        if not nodes:
            logging.error("No reference nodes available! returning empty string.")
            return ""

        # Flatten and sort nodes (descending by score)
        all_nodes: list[NodeWithScore] = sorted(
            (
                node
                for subq_node in nodes
                if subq_node is not None
                for node in subq_node.sources
                if node.metadata.get("url")
            ),
            key=lambda x: x.score,
            reverse=True,
        )

        # De-duplicate URLs, and filter by score threshold
        seen_urls = set()
        deduped_nodes = []
        for node in all_nodes:
            if node.score > self.threshold and node.metadata["url"] not in seen_urls:
                deduped_nodes.append(node)
                seen_urls.add(node.metadata["url"])

        # Take only top references up to max_references
        limited_nodes = deduped_nodes[: self.max_references]

        # If nothing remains after filtering, return empty
        if not limited_nodes:
            logging.error(
                f"All node scores are below threshold ({self.threshold}) "
                "or no valid URLs. Returning empty string!"
            )
            return ""

        # Format the sources
        sources_str = "\n".join(
            f"[{idx + 1}] {node.metadata['url']}"
            for idx, node in enumerate(limited_nodes)
        )
        return f"Top {min(len(limited_nodes), self.max_references)} references:\n{sources_str}"
