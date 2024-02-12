import os
import unittest
from unittest.mock import patch

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from llama_index.schema import NodeWithScore, TextNode
from sqlalchemy.exc import OperationalError
from utils.query_engine.level_based_platform_query_engine import (
    LevelBasedPlatformQueryEngine,
)


class TestLevelBasedPlatformQueryEngine(unittest.TestCase):
    def setUp(self):
        """
        Set up common parameters for testing
        """
        self.community_id = "test_community"
        self.level1_key = "channel"
        self.level2_key = "thread"
        self.platform_table_name = "discord"
        self.date_key = "date"
        os.environ["OPENAI_API_KEY"] = "sk-some_creds"

    def test_prepare_platform_engine(self):
        """
        Test prepare_platform_engine method with sample data
        """
        # the output should always have a `date` key for each dictionary
        filters = [
            {"channel": "general", "thread": "some_thread", "date": "2023-01-02"},
            {"channel": "general", "thread": "discussion", "date": "2024-01-03"},
            {"channel": "general#2", "thread": "Agenda", "date": "2022-01-01"},
        ]

        engine = LevelBasedPlatformQueryEngine.prepare_platform_engine(
            community_id=self.community_id,
            platform_table_name=self.platform_table_name,
            filters=filters,
            testing=True,
        )
        self.assertIsNotNone(engine)

    def test_prepare_engine_auto_filter_raise_error(self):
        """
        Test prepare_engine_auto_filter method with sample data
        when an error was raised
        """
        with patch.object(
            ForumBasedSummaryRetriever, "define_filters"
        ) as mock_retriever:
            # the output should always have a `date` key for each dictionary
            mock_retriever.return_value = [
                {"channel": "general", "thread": "some_thread", "date": "2023-01-02"},
                {"channel": "general", "thread": "discussion", "date": "2024-01-03"},
                {"channel": "general#2", "thread": "Agenda", "date": "2022-01-01"},
            ]

            with self.assertRaises(OperationalError):
                # no database with name of `test_community` is available
                _ = LevelBasedPlatformQueryEngine.prepare_engine_auto_filter(
                    community_id=self.community_id,
                    query="test query",
                    platform_table_name=self.platform_table_name,
                    level1_key=self.level1_key,
                    level2_key=self.level2_key,
                    date_key=self.date_key,
                )

    def test_prepare_engine_auto_filter(self):
        """
        Test prepare_engine_auto_filter method with sample data in normal condition
        """
        with patch.object(RetrieveSimilarNodes, "query_db") as mock_query:
            # the output should always have a `date` key for each dictionary
            mock_query.return_value = [
                NodeWithScore(
                    node=TextNode(
                        text="some summaries #1",
                        metadata={
                            "thread": "thread#1",
                            "channel": "channel#1",
                            "date": "2022-01-01",
                        },
                    ),
                    score=0,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="some summaries #2",
                        metadata={
                            "thread": "thread#3",
                            "channel": "channel#2",
                            "date": "2022-01-02",
                        },
                    ),
                    score=0,
                ),
            ]

            # no database with name of `test_community` is available
            engine = LevelBasedPlatformQueryEngine.prepare_engine_auto_filter(
                community_id=self.community_id,
                query="test query",
                platform_table_name=self.platform_table_name,
                level1_key=self.level1_key,
                level2_key=self.level2_key,
                date_key=self.date_key,
                include_summary_context=True,
            )
            self.assertIsNotNone(engine)
