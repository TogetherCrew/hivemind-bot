from unittest import TestCase

from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from utils.query_engine import GDriveQueryEngine
import pytest

@pytest.mark.skip(reason="Skipping test")
class TestGDriveQueryEngine(TestCase):
    def setUp(self) -> None:
        community_id = "sample_community"
        self.gdrive_query_engine = GDriveQueryEngine(community_id)

    def test_prepare_engine(self):
        gdrive_query_engine = self.gdrive_query_engine.prepare(testing=True)
        print(gdrive_query_engine.__dict__)
        self.assertIsInstance(gdrive_query_engine.retriever, VectorIndexRetriever)
