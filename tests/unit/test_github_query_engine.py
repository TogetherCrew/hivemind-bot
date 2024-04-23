from unittest import TestCase

from bot.retrievers.custom_retriever import CustomVectorStoreRetriever
from utils.query_engine import GitHubQueryEngine


class TestGitHubQueryEngine(TestCase):
    def setUp(self) -> None:
        community_id = "sample_community"
        self.github_query_engine = GitHubQueryEngine(community_id)

    def test_prepare_engine(self):
        github_query_engine = self.github_query_engine.prepare(testing=True)
        print(github_query_engine.__dict__)
        self.assertIsInstance(github_query_engine.retriever, CustomVectorStoreRetriever)
