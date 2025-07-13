import os
import unittest

from llama_index.core.base.base_query_engine import BaseQueryEngine
from utils.query_engine.prepare_discord_query_engine import prepare_discord_engine


class TestPrepareDiscordEngine(unittest.TestCase):
    def setUp(self):
        # Set up environment variables for testing
        os.environ["CHUNK_SIZE"] = "128"
        os.environ["EMBEDDING_DIM"] = "256"
        os.environ["K1_RETRIEVER_SEARCH"] = "20"
        os.environ["K2_RETRIEVER_SEARCH"] = "5"
        os.environ["D_RETRIEVER_SEARCH"] = "3"
        os.environ["OPENAI_API_KEY"] = "sk-some_creds"

    def test_prepare_discord_engine(self):
        community_id = "123456"
        platform_id = "123456"

        # Call the function
        query_engine = prepare_discord_engine(
            community_id,
            platform_id=platform_id,
            enable_answer_skipping=False,
            testing=True,
        )

        self.assertIsNotNone(query_engine)
        self.assertIsInstance(query_engine, BaseQueryEngine)
