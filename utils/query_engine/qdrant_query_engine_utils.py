from datetime import datetime, timedelta
from dateutil.parser import parse

from llama_index.core.schema import NodeWithScore
from schema.type import DataType
from qdrant_client.http import models


class QdrantEngineUtils:
    def __init__(
        self,
        metadata_date_key: str,
        metadata_date_format: DataType,
        date_margin: int,
    ) -> None:
        self.metadata_date_key = metadata_date_key
        self.metadata_date_format = metadata_date_format
        self.date_margin = date_margin

    def define_raw_data_filters(self, dates: list[str]) -> list[models.FieldCondition]:
        """
        define the filters to be applied on raw data given the dates

        Parameters
        -----------
        dates : list[str]
            a list of dates that should be a string. i.e. with format of `%Y-%m-%d`
            the date should be representing a day

        Returns
        ---------
        should_filters : list[models.FieldCondition]
            the filters to be applied on raw data
        """
        should_filters: set[models.FieldCondition] = set()
        expanded_dates: set[datetime] = set()

        # accounting for the date margin
        for date in dates:
            day_value = parse(date)
            expanded_dates.add(day_value)

            for i in range(1, self.date_margin + 1):
                expanded_dates.add(day_value - timedelta(days=i))
                expanded_dates.add(day_value + timedelta(days=i))

        for day_value in expanded_dates:
            next_day = day_value + timedelta(days=1)

            if self.metadata_date_format is DataType.INTEGER:
                gte_value = int(day_value.timestamp())
                lte_value = int(next_day.timestamp())
            elif self.metadata_date_format is DataType.FLOAT:
                gte_value = day_value.timestamp()
                lte_value = next_day.timestamp()
            else:
                raise ValueError(
                    "raw data metadata `date` shouldn't be anything other than FLOAT or INTEGER"
                )

            should_filters.add(
                models.FieldCondition(
                    key=self.metadata_date_key,
                    range=models.Range(
                        gte=gte_value,
                        lte=lte_value,
                    ),
                )
            )

        return list(should_filters)

    def combine_nodes_for_prompt(
        self,
        summary_nodes: list[NodeWithScore],
        raw_nodes: list[NodeWithScore],
    ) -> str:
        """
        Combines summary nodes with their corresponding raw nodes based on date matching.

        Parameters
        ----------
        summary_nodes : list[NodeWithScore]
            list of summary nodes containing metadata with 'date' in "%Y-%m-%d" format
            and 'text' field
        raw_nodes : list[NodeWithScore]
            list of raw nodes containing metadata with self.me as float timestamp
            and 'text' field

        Returns
        -------
        prompt : str
            A formatted prompt combining matched summary and raw texts
        """
        # Create a mapping of date to raw nodes for efficient lookup
        raw_nodes_by_date: dict[str, list[NodeWithScore]] = {}

        for raw_node in raw_nodes:
            timestamp = raw_node.metadata[self.metadata_date_key]
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

            if date_str not in raw_nodes_by_date:
                raw_nodes_by_date[date_str] = []
            raw_nodes_by_date[date_str].append(raw_node)

        # Build the combined prompt
        combined_sections = []

        for summary_node in summary_nodes:
            date = summary_node.metadata["date"]
            summary_text = summary_node.text

            summaries = summary_text.split("\n")
            summary_bullets = set(summaries)
            summary_bullets.remove("")

            section = f"date: {date}\n\nSummary:\n" + "\n".join(summary_bullets) + "\n"

            if date in raw_nodes_by_date:
                raw_texts = [node.text for node in raw_nodes_by_date[date]]
                section += "Messages:\n" + "\n".join(raw_texts)

            combined_sections.append(section)

        return "\n\n" + "\n\n".join(combined_sections)
