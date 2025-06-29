from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging

from dateutil.parser import parse
from llama_index.core.schema import NodeWithScore
from qdrant_client.http import models
from schema.type import DataType
from utils.globals import EXCLUDED_DATE_MARGIN


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

    def define_raw_data_filters(self, dates: list[str | float]) -> models.Filter:
        """
        define the filters to be applied on raw data given the dates

        Parameters
        -----------
        dates : list[str]
            a list of dates that should be a string. i.e. with format of `%Y-%m-%d`
            the date should be representing a day

        Returns
        ---------
        filter : models.Filter
            the filters to be applied on raw data
        """
        should_filters: list[models.FieldCondition] = []
        expanded_dates: set[datetime] = set()

        # Calculate the cutoff timestamp for excluding recent messages
        # we want messages older than EXCLUDED_DATE_MARGIN minutes ago
        # to avoid the question being included within the context (real-time data ingestion case)
        cutoff_datetime = datetime.now(tz=timezone.utc) - timedelta(minutes=EXCLUDED_DATE_MARGIN)
        cutoff_timestamp = cutoff_datetime.timestamp()

        # accounting for the date margin
        for date in dates:
            if isinstance(date, str):
                # Ensure we parse the date in UTC timezone to match our cutoff
                day_value = parse(date).replace(tzinfo=timezone.utc)
            elif isinstance(date, float):
                # if it was timestamp, convert to UTC
                day_value = datetime.fromtimestamp(date, tz=timezone.utc)
            else:
                raise ValueError(f"Type {type(date)} date is not supported!")

            expanded_dates.add(day_value)

            for i in range(1, self.date_margin + 1):
                expanded_dates.add(day_value - timedelta(days=i))
                expanded_dates.add(day_value + timedelta(days=i))

        logging.info(f"QueryEngineUtils: Expanded dates: {[d.strftime('%Y-%m-%d %H:%M:%S %Z') for d in expanded_dates]}")
        for day_value in expanded_dates:
            # Start of the day in UTC
            day_start = day_value.replace(hour=0, minute=0, second=0, microsecond=0)
            # Start of next day in UTC
            next_day = day_start + timedelta(days=1)

            if self.metadata_date_format == DataType.INTEGER:
                gte_value = int(day_start.timestamp())
                # Ensure we don't include messages newer than the cutoff
                lte_value = min(int(next_day.timestamp()), int(cutoff_timestamp))
            elif self.metadata_date_format == DataType.FLOAT:
                gte_value = day_start.timestamp()
                # Ensure we don't include messages newer than the cutoff
                lte_value = min(next_day.timestamp(), cutoff_timestamp)
            else:
                raise ValueError(
                    (
                        "raw data metadata `date` shouldn't be anything other than FLOAT or INTEGER"
                        f"! The Current given one is: {self.metadata_date_format}"
                    )
                )

            # Only add the filter if the date range is valid (gte <= lte)
            if gte_value <= lte_value:
                should_filters.append(
                    models.FieldCondition(
                        key=self.metadata_date_key,
                        range=models.Range(
                            gte=gte_value,
                            lte=lte_value,
                        ),
                    )
                )

        # Create the filter with both should conditions for date ranges
        # AND a global must condition as a safety net to ensure NO recent messages get through
        must_filters = [
            models.FieldCondition(
                key=self.metadata_date_key,
                range=models.Range(
                    lte=int(cutoff_timestamp) if self.metadata_date_format == DataType.INTEGER else cutoff_timestamp,
                ),
            )
        ]


        if should_filters:
            filter = models.Filter(should=should_filters, must=must_filters)
        else:
            # If no date ranges are valid, just use the global cutoff filter
            filter = models.Filter(must=must_filters)

        return filter

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
            date_str = (
                datetime.fromtimestamp(timestamp)
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%d")
            )

            if date_str not in raw_nodes_by_date:
                raw_nodes_by_date[date_str] = []
            raw_nodes_by_date[date_str].append(raw_node)

        # A summary could be separated into multiple nodes
        # combining them together
        combined_summaries: dict[str, str] = defaultdict(str)
        for summary_node in summary_nodes:
            date = summary_node.metadata["date"]
            summary_text = summary_node.text

            summaries = summary_text.split("\n")
            summary_bullets = set(summaries)
            if "" in summary_bullets:
                summary_bullets.remove("")
            combined_summaries[date] += "\n".join(summary_bullets)

        # Build the combined prompt
        combined_sections = []

        for date, summary_bullets in combined_summaries.items():
            section = f"Date: {date}\nSummary:\n" + summary_bullets + "\n\n"

            if date in raw_nodes_by_date:
                raw_texts = [node.text for node in raw_nodes_by_date[date]]
                section += "Messages:\n" + "\n".join(raw_texts)

            combined_sections.append(section)

        return "\n\n".join(combined_sections)
