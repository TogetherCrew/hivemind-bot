import unittest
import os
from unittest.mock import patch, Mock
from utils.query_engine.discord_query_engine import prepare_discord_engine
from llama_index.core import BaseQueryEngine
from llama_index.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters


class TestPrepareDiscordEngine(unittest.TestCase):
    def setUp(self):
        # Set up environment variables for testing
        os.environ["CHUNK_SIZE"] = "128"
        os.environ["EMBEDDING_DIM"] = "256"
        os.environ["K1_RETRIEVER_SEARCH"] = "20"
        os.environ["K2_RETRIEVER_SEARCH"] = "5"
        os.environ["D_RETRIEVER_SEARCH"] = "3"

    def test_prepare_discord_engine(self):
        community_id = "123456"
        thread_names = ["thread1", "thread2"]
        channel_names = ["channel1", "channel2"]
        days = ["2022-01-01", "2022-01-02"]

        # Call the function
        query_engine = prepare_discord_engine(
            community_id,
            thread_names,
            channel_names,
            days,
            testing=True,
        )

        # Assertions
        self.assertIsInstance(query_engine, BaseQueryEngine)

        expected_filter = MetadataFilters(
            filters=[
                ExactMatchFilter(key="thread", value="thread1"),
                ExactMatchFilter(key="thread", value="thread2"),
                ExactMatchFilter(key="channel", value="channel1"),
                ExactMatchFilter(key="channel", value="channel2"),
                ExactMatchFilter(key="date", value="2022-01-01"),
                ExactMatchFilter(key="date", value="2022-01-02"),
            ],
            condition=FilterCondition.OR,
        )

        self.assertEqual(query_engine.retriever._filters, expected_filter)
        # this is the secondary search, so K2 should be for this
        self.assertEqual(query_engine.retriever._similarity_top_k, 5)
