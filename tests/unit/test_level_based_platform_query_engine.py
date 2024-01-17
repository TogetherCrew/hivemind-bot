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
        self.engine = LevelBasedPlatformQueryEngine(
            level1_key=self.level1_key,
            level2_key=self.level2_key,
            platform_table_name=self.platform_table_name,
            date_key=self.date_key,
        )

    def test_prepare_platform_engine(self):
        """
        Test prepare_platform_engine method with sample data
        """
        level1_names = ["general"]
        level2_names = ["discussion"]
        days = ["2022-01-01"]
        query_engine = self.engine.prepare_platform_engine(
            community_id=self.community_id,
            level1_names=level1_names,
            level2_names=level2_names,
            days=days,
        )
        self.assertIsNotNone(query_engine)

    def test_prepare_engine_auto_filter(self):
        """
        Test prepare_engine_auto_filter method with sample data
        """
        with patch.object(
            ForumBasedSummaryRetriever, "retreive_metadata"
        ) as mock_retriever:
            mock_retriever.return_value = (["general"], ["discussion"], ["2022-01-01"])
            query_engine = self.engine.prepare_engine_auto_filter(
                community_id=self.community_id, query="test query"
            )
            self.assertIsNotNone(query_engine)

    def test_prepare_engine_auto_filter_with_d(self):
        """
        Test prepare_engine_auto_filter method with a specific value for d
        """
        with patch.object(
            ForumBasedSummaryRetriever, "retreive_metadata"
        ) as mock_retriever:
            mock_retriever.return_value = (["general"], ["discussion"], ["2022-01-01"])
            query_engine = self.engine.prepare_engine_auto_filter(
                community_id=self.community_id,
                query="test query",
                d=7,  # Use a specific value for d
            )
            self.assertIsNotNone(query_engine)

    def test_prepare_engine_auto_filter_with_similarity_top_k(self):
        """
        Test prepare_engine_auto_filter method with a specific value for similarity_top_k
        """
        with patch.object(
            ForumBasedSummaryRetriever, "retreive_metadata"
        ) as mock_retriever:
            mock_retriever.return_value = (["general"], ["discussion"], ["2022-01-01"])
            query_engine = self.engine.prepare_engine_auto_filter(
                community_id=self.community_id,
                query="test query",
                similarity_top_k=10,  # Use a specific value for similarity_top_k
            )
            self.assertIsNotNone(query_engine)
