import os
import unittest
from unittest.mock import patch

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
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
            {"channel": "general", "date": "2023-01-02"},
            {"thread": "discussion", "date": "2024-01-03"},
            {"date": "2022-01-01"},
        ]

        engine = LevelBasedPlatformQueryEngine.prepare_platform_engine(
            community_id=self.community_id,
            platform_table_name=self.platform_table_name,
            filters=filters,
            testing=True,
        )
        self.assertIsNotNone(engine)

    def test_prepare_engine_auto_filter(self):
        """
        Test prepare_engine_auto_filter method with sample data
        """
        with patch.object(
            ForumBasedSummaryRetriever, "retreive_filtering"
        ) as mock_retriever:
            # the output should always have a `date` key for each dictionary
            mock_retriever.return_value = [
                {"channel": "general", "date": "2023-01-02"},
                {"thread": "discussion", "date": "2024-01-03"},
                {"date": "2022-01-01"},
            ]

            engine = LevelBasedPlatformQueryEngine.prepare_engine_auto_filter(
                community_id=self.community_id,
                query="test query",
                platform_table_name=self.platform_table_name,
                level1_key=self.level1_key,
                level2_key=self.level2_key,
                date_key=self.date_key,
            )
            self.assertIsNotNone(engine)
