from unittest import TestCase
from unittest.mock import MagicMock

from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from llama_index.schema import TextNode


class TestRetrieveSimilarNodes(TestCase):
    def setUp(self):
        self.table_name = "sample_table"
        self.dbname = "community_some_id"

        self.vector_store = MagicMock()
        self.embed_model = MagicMock()
        self.retriever = RetrieveSimilarNodes(
            vector_store=self.vector_store,
            similarity_top_k=5,
            embed_model=self.embed_model,
        )

    def test_init(self):
        self.assertEqual(self.retriever._similarity_top_k, 5)
        self.assertEqual(self.vector_store, self.retriever._vector_store)

    def test_get_nodes_with_score(self):
        # Test the _get_nodes_with_score private method
        query_result = MagicMock()
        query_result.nodes = [TextNode(), TextNode(), TextNode()]
        query_result.similarities = [0.8, 0.9, 0.7]

        result = self.retriever._get_nodes_with_score(query_result)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0].score, 0.8, delta=0.001)
