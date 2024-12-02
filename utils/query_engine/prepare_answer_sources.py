import logging

from llama_index.core.schema import NodeWithScore


class PrepareAnswerSources:
    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def prepare_answer_sources(self, nodes: list[NodeWithScore]) -> str:
        """
        Prepares a formatted string containing source URLs from the provided nodes.

        This method extracts URLs from the metadata of each node and combines them
        into a newline-separated string. Only nodes with valid URLs in their metadata
        are included in the output.

        Parameters
        ------------
        nodes : list[NodeWithScore]
            A list of nodes that was used for answering a question. Each node
            should have a metadata attribute containing an optional 'url' field.


        Returns
        -------
        all_sources : str
            A newline-separated string of source URLs. Returns an empty string if
            no valid URLs are found in the nodes' metadata.
        """
        if len(nodes) == 0:
            logging.error("No reference nodes available! returning empty string.")
            return ""

        # link of places that we got the answer from
        all_sources: str = "References:\n"
        for tool_nodes in nodes:
            # platform name
            tool_name = tool_nodes.sub_q.tool_name
            urls = [
                node.metadata.get("url")
                for node in tool_nodes.sources
                if node.score >= self.threshold and node.metadata.get("url") is not None
            ]
            if urls:
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
