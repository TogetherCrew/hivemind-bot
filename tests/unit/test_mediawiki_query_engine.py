from unittest import TestCase

from bot.retrievers.custom_retriever import CustomVectorStoreRetriever
from utils.query_engine import MediaWikiQueryEngine


class TestMediaWikiQueryEngine(TestCase):
    def setUp(self) -> None:
        community_id = "sample_community"
        self.notion_query_engine = MediaWikiQueryEngine(community_id)

    def test_prepare_engine(self):
        notion_query_engine = self.notion_query_engine.prepare(testing=True)
        print(notion_query_engine.__dict__)
        self.assertIsInstance(notion_query_engine.retriever, CustomVectorStoreRetriever)
