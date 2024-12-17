import logging

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

    def prepare_answer_sources(self, nodes: list[NodeWithScore]) -> str:
        """
        Prepares a formatted string containing source URLs organized by tool name from the provided nodes.

        This method processes a list of nodes, filtering them based on a score threshold and
        organizing the URLs by their associated tool names. It creates a formatted output with
        URLs numbered under their respective tool sections, limiting the number of references
        per data source.

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
        str
            A formatted string containing numbered URLs organized by tool name, with the format:
            References:
            {tool_name}:
            [1] {url1}
            [2] {url2}

            Returns an empty string if:
            - The input nodes list is empty
            - No nodes meet the score threshold
            - No valid URLs are found in the nodes' metadata

        Notes
        -----
        - URLs are only included if their node's score meets or exceeds the threshold
          (default: 0.5)
        - Each tool's sources are grouped together and prefixed with the tool name
        - URLs are numbered sequentially within each tool's section
        - Maximum number of references per data source is limited by max_refs_per_source
          (default: 3)
        - References are selected based on highest scores when limiting
        - Logs error messages when no nodes are available or when all nodes are below
          the threshold
        """
        if len(nodes) == 0:
            logging.error("No reference nodes available! returning empty string.")
            return ""

        # link of places that we got the answer from
        all_sources: str = "References:\n"
        for tool_nodes in nodes:
            # platform name
            tool_name = tool_nodes.sub_q.tool_name

            # Filter and sort nodes by score
            valid_nodes = [
                node
                for node in tool_nodes.sources
                if node.score >= self.threshold and node.metadata.get("url") is not None
            ]
            valid_nodes.sort(key=lambda x: x.score, reverse=True)

            # Limit the number of references
            limited_nodes = valid_nodes[: self.max_refs_per_source]

            if limited_nodes:
                urls = [node.metadata.get("url") for node in limited_nodes]
                sources: list[str] = [
                    f"[{idx + 1}] {url}" for idx, url in enumerate(urls)
                ]
                sources_combined: str = "\n".join(sources)
                all_sources += f"{tool_name}:\n{sources_combined}\n\n"

        if all_sources == "References:\n":
            logging.error(
                f"All node scores are below threshold. threshold: {self.threshold}"
                ". returning empty string!"
            )
            return ""

        return all_sources.removesuffix("\n\n")
