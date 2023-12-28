from datetime import timedelta
from functools import partial
from unittest import TestCase
from unittest.mock import MagicMock

from retrievers.forum_summary_retriever import (
    ForumBasedSummaryRetriever,
)
from dateutil import parser
from llama_index import Document, MockEmbedding, ServiceContext, VectorStoreIndex


class TestDiscordSummaryRetriever(TestCase):
    def test_initialize_class(self):
        ForumBasedSummaryRetriever._setup_index = MagicMock()
        documents: list[Document] = []
        all_dates: list[str] = []

        for i in range(30):
            date = parser.parse("2023-08-01") + timedelta(days=i)
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

        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=mock_embedding_model()
        )
        ForumBasedSummaryRetriever._setup_index.return_value = (
            VectorStoreIndex.from_documents(
                documents=[doc], service_context=service_context
            )
        )

        base_summary_search = ForumBasedSummaryRetriever(
            table_name="sample",
            dbname="sample",
            embedding_model=mock_embedding_model(),
        )
        channels, threads, dates = base_summary_search.retreive_metadata(
            query="what is samplesample?",
            similarity_top_k=5,
            metadata_group1_key="channel",
            metadata_group2_key="thread",
            metadata_date_key="date",
        )
        self.assertIsInstance(threads, set)
        self.assertIsInstance(channels, set)
        self.assertIsInstance(dates, set)

        self.assertTrue(
            threads.issubset(
                set(
                    [
                        "thread0",
                        "thread1",
                        "thread2",
                        "thread3",
                        "thread4",
                    ]
                )
            )
        )
        self.assertTrue(
            channels.issubset(
                set(
                    [
                        "channel0",
                        "channel1",
                        "channel2",
                    ]
                )
            )
        )
        self.assertTrue(dates.issubset(all_dates))
