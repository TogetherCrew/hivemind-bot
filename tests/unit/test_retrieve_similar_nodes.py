from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

from bot.retrievers.retrieve_similar_nodes import RetrieveSimilarNodes
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import NodeWithScore, TextNode


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

    @patch.object(PGVectorStore, "_initialize")
    @patch.object(PGVectorStore, "_session")
    def test_query_db_with_filters_and_date(self, mock_session, mock_initialize):
        # Mock vector store initialization
        mock_initialize.return_value = None
        mock_session.begin = MagicMock()
        mock_session.execute = MagicMock()
        mock_session.execute.return_value = [1]

        vector_store = PGVectorStore.from_params(
            database="sample_db",
            host="sample_host",
            password="pass",
            port=5432,
            user="user",
            table_name=self.table_name,
            embed_dim=1536,
        )
        retrieve_similar_nodes = RetrieveSimilarNodes(vector_store, similarity_top_k=5)

        query = "test query"
        filters = [{"date": "2024-04-09"}]
        date_interval = 2  # Look for nodes within 2 days of the filter date

        # Call the query_db method with filters and date
        results = retrieve_similar_nodes.query_db(query, filters, date_interval)

        mock_initialize.assert_called_once()
        mock_session.assert_called_once()

        # Assert that the returned results are of type NodeWithScore
        self.assertTrue(isinstance(result, NodeWithScore) for result in results)

    @patch.object(PGVectorStore, "_initialize")
    @patch.object(PGVectorStore, "_session")
    def test_query_db_with_filters_and_date_aggregate_records(
        self, mock_session, mock_initialize
    ):
        # Mock vector store initialization
        mock_initialize.return_value = None
        mock_session.begin = MagicMock()
        mock_session.execute = MagicMock()
        mock_session.execute.return_value = [1]

        vector_store = PGVectorStore.from_params(
            database="sample_db",
            host="sample_host",
            password="pass",
            port=5432,
            user="user",
            table_name=self.table_name,
            embed_dim=1536,
        )
        retrieve_similar_nodes = RetrieveSimilarNodes(vector_store, similarity_top_k=5)

        query = "test query"
        filters = [{"date": "2024-04-09"}]
        date_interval = 2  # Look for nodes within 2 days of the filter date

        # Call the query_db method with filters and date
        results = retrieve_similar_nodes.query_db(
            query,
            filters,
            date_interval,
            aggregate_records=True,
            group_by_metadata=["thread"],
        )

        mock_initialize.assert_called_once()
        mock_session.assert_called_once()

        # Assert that the returned results are of type NodeWithScore
        self.assertTrue(isinstance(result, NodeWithScore) for result in results)
