import os
import unittest
from unittest.mock import patch

from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from llama_index.core.schema import NodeWithScore, TextNode
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

    def test_prepare_context_str_without_summaries(self):
        """
        test prepare the context string while not having the summaries nodes
        """
        with patch.object(RetrieveSimilarNodes, "query_db") as mock_query:
            summary_nodes = []
            mock_query.return_value = summary_nodes

            engine = LevelBasedPlatformQueryEngine.prepare_engine_auto_filter(
                community_id=self.community_id,
                query="test query",
                platform_table_name=self.platform_table_name,
                level1_key=self.level1_key,
                level2_key=self.level2_key,
                date_key=self.date_key,
                include_summary_context=True,
            )

            raw_nodes = [
                NodeWithScore(
                    node=TextNode(
                        text="content1",
                        metadata={
                            "author_username": "user1",
                            "channel": "channel#1",
                            "thread": "thread#1",
                            "date": "2022-01-01",
                        },
                    ),
                    score=0,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="content2",
                        metadata={
                            "author_username": "user2",
                            "channel": "channel#2",
                            "thread": "thread#3",
                            "date": "2022-01-02",
                        },
                    ),
                    score=0,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="content4",
                        metadata={
                            "author_username": "user3",
                            "channel": "channel#2",
                            "thread": "thread#3",
                            "date": "2022-01-02",
                        },
                    ),
                    score=0,
                ),
            ]

            contest_str = engine._prepare_context_str(raw_nodes, summary_nodes)
            expected_context_str = (
                "author: user1\n"
                "message_date: 2022-01-01\n"
                "message 1: content1\n\n"
                "author: user2\n"
                "message_date: 2022-01-02\n"
                "message 2: content2\n\n"
                "author: user3\n"
                "message_date: 2022-01-02\n"
                "message 3: content4\n"
            )
            self.assertEqual(contest_str, expected_context_str)

    def test_prepare_context_str_with_summaries(self):
        """
        test prepare the context string having the summaries nodes
        """

        with patch.object(RetrieveSimilarNodes, "query_db") as mock_query:
            summary_nodes = [
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
            mock_query.return_value = summary_nodes

            engine = LevelBasedPlatformQueryEngine.prepare_engine_auto_filter(
                community_id=self.community_id,
                query="test query",
                platform_table_name=self.platform_table_name,
                level1_key=self.level1_key,
                level2_key=self.level2_key,
                date_key=self.date_key,
                include_summary_context=True,
            )

            raw_nodes = [
                NodeWithScore(
                    node=TextNode(
                        text="content1",
                        metadata={
                            "author_username": "user1",
                            "channel": "channel#1",
                            "thread": "thread#1",
                            "date": "2022-01-01",
                        },
                    ),
                    score=0,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="content2",
                        metadata={
                            "author_username": "user2",
                            "channel": "channel#2",
                            "thread": "thread#3",
                            "date": "2022-01-02",
                        },
                    ),
                    score=0,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="content4",
                        metadata={
                            "author_username": "user3",
                            "channel": "channel#2",
                            "thread": "thread#3",
                            "date": "2022-01-02",
                        },
                    ),
                    score=0,
                ),
            ]

            contest_str = engine._prepare_context_str(raw_nodes, summary_nodes)
            expected_context_str = (
                "channel: channel#1\n"
                "thread: thread#1\n"
                "date: 2022-01-01\n"
                "summary: some summaries #1\n"
                "messages:\n"
                "  author: user1\n"
                "  message_date: 2022-01-01\n"
                "  message 1: content1\n\n"
                "channel: channel#2\n"
                "thread: thread#3\n"
                "date: 2022-01-02\n"
                "summary: some summaries #2\n"
                "messages:\n"
                "  author: user2\n"
                "  message_date: 2022-01-02\n"
                "  message 1: content2\n\n"
                "  author: user3\n"
                "  message_date: 2022-01-02\n"
                "  message 2: content4\n\n"
            )
            self.assertEqual(contest_str, expected_context_str)
