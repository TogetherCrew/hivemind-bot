import os
import unittest

from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters
from utils.query_engine.discourse_query_engine import prepare_discourse_engine


class TestPrepareDiscourseEngine(unittest.TestCase):
    def setUp(self):
        # Set up environment variables for testing
        os.environ["CHUNK_SIZE"] = "128"
        os.environ["EMBEDDING_DIM"] = "256"
        os.environ["K1_RETRIEVER_SEARCH"] = "20"
        os.environ["K2_RETRIEVER_SEARCH"] = "5"
        os.environ["D_RETRIEVER_SEARCH"] = "3"

    def test_prepare_discourse_engine(self):
        community_id = "123456"
        topic_names = ["topic1", "topic2"]
        category_names = ["category1", "category2"]
        days = ["2022-01-01", "2022-01-02"]

        # Call the function
        query_engine = prepare_discourse_engine(
            community_id=community_id,
            category_names=category_names,
            topic_names=topic_names,
            days=days,
            testing=True,
        )

        # Assertions
        self.assertIsInstance(query_engine, BaseQueryEngine)

        expected_filter = MetadataFilters(
            filters=[
                ExactMatchFilter(key="category", value="category1"),
                ExactMatchFilter(key="category", value="category2"),
                ExactMatchFilter(key="topic", value="topic1"),
                ExactMatchFilter(key="topic", value="topic2"),
                ExactMatchFilter(key="date", value="2022-01-01"),
                ExactMatchFilter(key="date", value="2022-01-02"),
            ],
            condition=FilterCondition.OR,
        )

        self.assertEqual(query_engine.retriever._filters, expected_filter)
        # this is the secondary search, so K2 should be for this
        self.assertEqual(query_engine.retriever._similarity_top_k, 5)
