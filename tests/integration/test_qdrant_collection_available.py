from unittest import TestCase

from qdrant_client import models
from tc_hivemind_backend.db.qdrant import QdrantSingleton
from utils.qdrant_utils import QDrantUtils


class TestQDrantAvailableCollection(TestCase):
    def setUp(self) -> None:
        self.community_id = "community_sample"
        self.qdrant_client = QdrantSingleton.get_instance().get_client()
        self.qdrant_utils = QDrantUtils(self.community_id)

        # deleting all collections
        collections = self.qdrant_client.get_collections()
        for col in collections.collections:
            self.qdrant_client.delete_collection(col.name)

    def test_no_collection_available(self):
        platform = "platform1"
        available = self.qdrant_utils.check_collection_exist(platform)

        self.assertIsInstance(available, bool)
        self.assertFalse(available)

    def test_single_collection_available(self):
        platform = "platform1"
        collection_name = f"{self.community_id}_{platform}"
        self.qdrant_client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=100, distance=models.Distance.COSINE
            ),
        )
        available = self.qdrant_utils.check_collection_exist(platform)

        self.assertIsInstance(available, bool)
        self.assertTrue(available)

    def test_multiple_collections_but_not_input(self):
        """
        test if there was multiple collections available
        but it isn't the collection we want to check for
        """
        platforms = ["platform1", "platform2", "platform3"]
        for plt in platforms:
            collection_name = f"{self.community_id}_{plt}"
            self.qdrant_client.create_collection(
                collection_name,
                vectors_config=models.VectorParams(
                    size=100, distance=models.Distance.COSINE
                ),
            )

        available = self.qdrant_utils.check_collection_exist("platform4")

        self.assertIsInstance(available, bool)
        self.assertFalse(available)

    def test_multiple_collections_available_given_input(self):
        """
        test multiple collections available with given input
        """
        platforms = ["platform1", "platform2", "platform3"]
        for plt in platforms:
            collection_name = f"{self.community_id}_{plt}"
            self.qdrant_client.create_collection(
                collection_name,
                vectors_config=models.VectorParams(
                    size=100, distance=models.Distance.COSINE
                ),
            )

        available = self.qdrant_utils.check_collection_exist(
            platforms[0],
        )

        self.assertIsInstance(available, bool)
        self.assertTrue(available)
