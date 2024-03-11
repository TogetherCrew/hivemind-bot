from datetime import timedelta
from functools import partial
from unittest import TestCase
from unittest.mock import MagicMock

from bot.retrievers.forum_summary_retriever import ForumBasedSummaryRetriever
from dateutil import parser
from llama_index.core import Document, MockEmbedding, Settings, VectorStoreIndex


class TestDiscordSummaryRetriever(TestCase):
    def test_initialize_class(self):
        ForumBasedSummaryRetriever._setup_index = MagicMock()
        documents: list[Document] = []
        all_dates: list[str] = []

        start_date = parser.parse("2023-08-01")
        for i in range(30):
            date = start_date + timedelta(days=i)
            doc_date = date.strftime("%Y-%m-%d")
            doc = Document(
                text="SAMPLESAMPLESAMPLE",
                metadata={
                    "thread": f"thread{i % 5}",
                    "channel": f"channel{i % 3}",
                    "date": doc_date,
                },
            )
            all_dates.append(doc_date)
            documents.append(doc)

        mock_embedding_model = partial(MockEmbedding, embed_dim=1024)

        Settings.llm = None
        Settings.embed_model = mock_embedding_model()
        ForumBasedSummaryRetriever._setup_index.return_value = (
            VectorStoreIndex.from_documents(documents=[doc])
        )

        base_summary_search = ForumBasedSummaryRetriever(
            table_name="sample",
            dbname="sample",
            embedding_model=mock_embedding_model(),
        )
        filters = base_summary_search.retreive_filtering(
            query="what is samplesample?",
            similarity_top_k=5,
            metadata_group1_key="channel",
            metadata_group2_key="thread",
            metadata_date_key="date",
        )

        self.assertIsInstance(filters, list)

        expected_dates = [
            (start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)
        ]
        for filter in filters:
            self.assertIsInstance(filter, dict)
            self.assertIn(
                filter["thread"],
                [
                    "thread0",
                    "thread1",
                    "thread2",
                    "thread3",
                    "thread4",
                ],
            )
            self.assertIn(filter["channel"], ["channel0", "channel1", "channel2"])
            date = parser.parse("2023-08-01") + timedelta(days=i)
            doc_date = date.strftime("%Y-%m-%d")
            self.assertIn(filter["date"], expected_dates)
