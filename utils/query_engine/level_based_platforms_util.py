import logging

from dateutil import parser
from llama_index.schema import NodeWithScore


class LevelBasedPlatformUtils:
    def __init__(self, level1_key: str, level2_key: str, date_key: str) -> None:
        self.level1_key = level1_key
        self.level2_key = level2_key
        self.date_key = date_key

    def prepare_prompt_with_metadata_info(
        self, nodes: list[NodeWithScore], prefix: str = ""
    ) -> str:
        """
        prepare a prompt with given nodes including the nodes' metadata
        Note: the prefix is set before each text!
        """
        context_str = "\n".join(
            [
                prefix
                + "author: "
                + node.metadata["author_username"]
                + "\n"
                + prefix
                + "message_date: "
                + node.metadata["date"]
                + "\n"
                + prefix
                + f"message {idx + 1}: "
                + node.get_content()
                + "\n"
                for idx, node in enumerate(nodes)
            ]
        )

        return context_str

    def group_nodes_per_metadata(
        self,
        nodes: list[NodeWithScore],
    ) -> dict[str, dict[str, dict[str, list[NodeWithScore]]]]:
        """
        group all nodes based on their level1 and level2 metadata

        Parameters
        -----------
        nodes : list[NodeWithScore]
            a list of raw nodes

        Returns
        ---------
        grouped_nodes : dict[str, dict[str, dict[str, list[NodeWithScore]]]]
            a list of nodes grouped by
            - `level1_key`
            - `level2_key`
            - and the last dict key `date_key`

            The values of the nested dictionary are the nodes grouped
        """
        grouped_nodes: dict[str, dict[str, dict[str, list[NodeWithScore]]]] = {}
        for node in nodes:
            level1_title = node.metadata[self.level1_key]
            level2_title = node.metadata[self.level2_key]
            date_str = node.metadata[self.date_key]
            date = parser.parse(date_str).strftime("%Y-%m-%d")

            # defining an empty list (if keys weren't previously made)
            grouped_nodes.setdefault(level1_title, {}).setdefault(
                level2_title, {}
            ).setdefault(date, [])
            # Adding to list
            grouped_nodes[level1_title][level2_title][date].append(node)

        return grouped_nodes

    def prepare_context_str_based_on_summaries(
        self,
        grouped_raw_nodes: dict[str, dict[str, dict[str, list[NodeWithScore]]]],
        grouped_summary_nodes: dict[str, dict[str, dict[str, list[NodeWithScore]]]],
    ) -> tuple[
        str,
        tuple[
            list[dict[str, str | None]],
            dict[str, dict[str, dict[str, list[NodeWithScore]]]],
        ],
    ]:
        """
        prepare prompt context having the summaries within it
        """
        context_str: str = ""

        summary_nodes_to_fetch_filters: list[dict[str, str | None]] = []
        # in case of summary wasn't available for them
        raw_nodes_missed: dict[str, dict[str, dict[str, list[NodeWithScore]]]] = {}

        for level1_title in grouped_raw_nodes:
            for level2_title in grouped_raw_nodes[level1_title]:
                for date in grouped_raw_nodes[level1_title][level2_title]:
                    raw_nodes = grouped_raw_nodes[level1_title][level2_title][date]

                    # the summary_nodes should be always 0 or 1 node
                    summary_nodes = (
                        grouped_summary_nodes.get(level1_title, {})
                        .get(level2_title, {})
                        .get(date, [])
                    )
                    if len(summary_nodes) == 1:
                        logging.debug(
                            f"{len(raw_nodes)} messages available for "
                            f"{self.level1_key}: {level1_title}, "
                            f"{self.level2_key}: {level2_title}, "
                            f"{self.date_key}: {date}"
                        )
                        summary_node = summary_nodes[0]

                        if level1_title != "None" and level2_title != "None":
                            node_context: str = (
                                f"{self.level1_key}: {level1_title}\n"
                                f"{self.level2_key}: {level2_title}\n"
                                f"{self.date_key}: {date}\n"
                                f"summary: {summary_node.text}\n"
                                "messages:\n"
                            )
                        elif level1_title == "None":
                            node_context: str = (
                                f"{self.level1_key}: main {self.level2_key}\n"
                                f"{self.level2_key}: {level2_title}\n"
                                f"{self.date_key}: {date}\n"
                                f"summary: {summary_node.text}\n"
                                "messages:\n"
                            )
                        elif level2_title == "None":
                            node_context: str = (
                                f"{self.level1_key}: {level1_title}\n"
                                f"{self.level2_key}: main {self.level1_key}\n"
                                f"{self.date_key}: {date}\n"
                                f"summary: {summary_node.text}\n"
                                "messages:\n"
                            )
                        node_context += self.prepare_prompt_with_metadata_info(
                            raw_nodes, prefix="  "
                        )

                        context_str += node_context + "\n"
                    elif len(summary_nodes) == 0:
                        logging.info(
                            "No summary messages available for "
                            f"{self.level1_key}: {level1_title}, "
                            f"{self.level2_key}: {level2_title}, "
                            f"{self.date_key}: {date}"
                            "\t will fetch them after"
                        )
                        summary_nodes_to_fetch_filters.append(
                            {
                                self.level1_key: level1_title,
                                self.level2_key: level2_title,
                                self.date_key: date,
                                # we need the thread summaries
                                "type": "thread",
                            }
                        )
                        raw_nodes_missed.setdefault(level1_title, {}).setdefault(
                            level2_title, {}
                        ).setdefault(date, [])
                        raw_nodes_missed[level1_title][level2_title][date].extend(
                            raw_nodes
                        )
                    else:
                        logging.info(f"len(summary_nodes) {len(summary_nodes)}")
                        raise ValueError(
                            "Not possible to have multiple summaries for a"
                            f" combination of "
                            f"{self.level1_key}-{self.level2_key}-{self.date_key}"
                        )

        return context_str, (summary_nodes_to_fetch_filters, raw_nodes_missed)
