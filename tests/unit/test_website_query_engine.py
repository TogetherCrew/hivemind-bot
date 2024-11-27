from unittest import TestCase

from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from utils.query_engine import WebsiteQueryEngine


class TestNotionQueryEngine(TestCase):
    def setUp(self) -> None:
        community_id = "sample_community"
        self.website_query_engine = WebsiteQueryEngine(community_id)

    def test_prepare_engine(self):
        website_query_engine = self.website_query_engine.prepare(testing=True)
        print(website_query_engine.__dict__)
        self.assertIsInstance(website_query_engine.retriever, VectorIndexRetriever)
