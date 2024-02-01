from functools import partial
from unittest import TestCase
from unittest.mock import MagicMock

from bot.retrievers.summary_retriever_base import BaseSummarySearch
from llama_index import Document, MockEmbedding, ServiceContext, VectorStoreIndex
from llama_index.schema import NodeWithScore


class TestSummaryRetrieverBase(TestCase):
    def test_initialize_class(self):
        BaseSummarySearch._setup_index = MagicMock()
        doc = Document(text="SAMPLESAMPLESAMPLE")
        mock_embedding_model = partial(MockEmbedding, embed_dim=1024)

        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=mock_embedding_model()
        )
        BaseSummarySearch._setup_index.return_value = VectorStoreIndex.from_documents(
            documents=[doc], service_context=service_context
        )

        base_summary_search = BaseSummarySearch(
            table_name="sample",
            dbname="sample",
            embedding_model=mock_embedding_model(),
        )
        nodes = base_summary_search.get_similar_nodes(query="what is samplesample?")
        self.assertIsInstance(nodes, list)
        self.assertIsInstance(nodes[0], NodeWithScore)

    def test_setup_index(self):
        table_name = "your_table_name"
        dbname = "your_db_name"
        embedding_model = MagicMock()
        search_instance = BaseSummarySearch(table_name, dbname, embedding_model)

        index = search_instance._setup_index(
            table_name, dbname, embedding_model, testing=True
        )
        self.assertIsNotNone(index)
        self.assertIsInstance(index, VectorStoreIndex)
