import logging
from collections import defaultdict

from llama_index.core.schema import NodeWithScore
from utils.globals import REFERENCE_SCORE_THRESHOLD


class PrepareAnswerSources:
    def __init__(
        self, threshold: float = REFERENCE_SCORE_THRESHOLD, max_refs_per_source: int = 3
    ) -> None:
        """
        Initialize the PrepareAnswerSources class.

        Parameters
        ----------
        threshold : float, optional
            Minimum score threshold for including a node's URL, by default 0.5 set in globals file
        max_refs_per_source : int, optional
            Maximum number of references to include per data source, by default 3
        """
        self.threshold = threshold
        self.max_refs_per_source = max_refs_per_source

    def prepare_answer_sources(self, nodes: list[NodeWithScore | None]) -> str:
        """
        Prepares a formatted string containing unique source URLs organized by tool name
        from the provided nodes, avoiding duplicate URLs.

        Parameters
        ----------
        nodes : list[NodeWithScore]
            A list of node collections used for answering a question. Each node collection
            contains:
            - sub_q.tool_name: Name of the tool that generated these nodes
            - sources: List of nodes, each containing:
                - score: Relevance score of the node
                - metadata: Dictionary containing an optional 'url' field

        Returns
        -------
        all_sources : str
            A formatted string containing numbered URLs organized by tool name, with the format:
            References:
            {tool_name}:
            [1] {url1}
            [2] {url2}

            Returns an empty string if:
            - The input nodes list is empty
            - No nodes meet the score threshold
            - No valid URLs are found in the nodes' metadata
        """
        if len(nodes) == 0:
            logging.error("No reference nodes available! returning empty string.")
            return ""

        cleaned_nodes = [n for n in nodes if n is not None]

        # Group nodes by tool name while filtering by score and valid URL
        tool_sources = defaultdict(list)
        for tool_nodes in cleaned_nodes:
            tool_name = tool_nodes.sub_q.tool_name
            for node in tool_nodes.sources:
                if (
                    node.score >= self.threshold
                    and node.metadata.get("url") is not None
                ):
                    tool_sources[tool_name].append(node)

        if not tool_sources:
            logging.error(
                f"All node scores are below threshold ({self.threshold}). Returning empty string!"
            )
            return ""

        all_sources = "References:\n"

        # Process each tool's nodes, remove duplicate URLs, sort and limit references
        for tool_name, nodes_list in tool_sources.items():
            unique_nodes = {}
            for node in nodes_list:
                url = node.metadata.get("url")
                # If URL not seen yet or current node has a higher score, update it
                if url not in unique_nodes or node.score > unique_nodes[url].score:
                    unique_nodes[url] = node

            # Sort nodes by score in descending order and limit references
            sorted_nodes = sorted(
                unique_nodes.values(), key=lambda x: x.score, reverse=True
            )
            limited_nodes = sorted_nodes[: self.max_refs_per_source]

            if limited_nodes:
                sources = [
                    f"[{idx + 1}] {node.metadata['url']}"
                    for idx, node in enumerate(limited_nodes)
                ]
                sources_combined = "\n".join(sources)
                all_sources += f"{tool_name}:\n{sources_combined}\n\n"

        if all_sources == "References:\n":
            logging.error(
                f"All node scores are below threshold ({self.threshold}). Returning empty string!"
            )
            return ""

        # Remove trailing newlines
        return all_sources.removesuffix("\n\n")
