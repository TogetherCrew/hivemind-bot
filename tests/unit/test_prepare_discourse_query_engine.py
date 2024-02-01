import os
import unittest

from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters
from utils.query_engine.prepare_discourse_query_engine import prepare_discourse_engine


class TestPrepareDiscourseEngine(unittest.TestCase):
    def setUp(self):
        # Set up environment variables for testing
        os.environ["CHUNK_SIZE"] = "128"
        os.environ["EMBEDDING_DIM"] = "256"
        os.environ["K1_RETRIEVER_SEARCH"] = "20"
        os.environ["K2_RETRIEVER_SEARCH"] = "5"
        os.environ["D_RETRIEVER_SEARCH"] = "3"
        os.environ["OPENAI_API_KEY"] = "sk-some_creds"

    def test_prepare_discourse_engine(self):
        community_id = "123456"
        filters = [
            {"category": "general", "date": "2023-01-02"},
            {"topic": "discussion", "date": "2024-01-03"},
            {"date": "2022-01-01"},
        ]

        # Call the function
        query_engine = prepare_discourse_engine(
            community_id=community_id,
            filters=filters,
            testing=True,
        )

        self.assertIsNotNone(query_engine)
        self.assertIsInstance(query_engine, BaseQueryEngine)
