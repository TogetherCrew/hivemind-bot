from unittest import TestCase

from qdrant_client import models
from tc_hivemind_backend.db.qdrant import QdrantSingleton
from utils.qdrant_utils import QDrantUtils


class TestQDrantAvailableCollection(TestCase):
    def setUp(self) -> None:
        self.qdrant_client = QdrantSingleton.get_instance().get_client()
        collections = self.qdrant_client.get_collections()
        for col in collections.collections:
            self.qdrant_client.delete_collection(col.name)
        self.qdrant_utils = QDrantUtils()

    def test_no_collection_available(self):
        collection_name = "sample_collection"
        available = self.qdrant_utils.chech_collection_exist(collection_name)

        self.assertIsInstance(available, bool)
        self.assertFalse(available)

    def test_single_collection_available(self):
        collection_name = "sample_collection"
        self.qdrant_client.delete_collection(collection_name)
        self.qdrant_client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=100, distance=models.Distance.COSINE
            ),
        )
        available = self.qdrant_utils.chech_collection_exist(collection_name)

        self.assertIsInstance(available, bool)
        self.assertTrue(available)

    def test_multiple_collections_but_not_input(self):
        """
        test if there was multiple collections available
        but it isn't the collection we want to check for
        """
        collections = ["collection1", "collection2", "collection3"]
        for col in collections:
            self.qdrant_client.create_collection(
                col,
                vectors_config=models.VectorParams(
                    size=100, distance=models.Distance.COSINE
                ),
            )

        available = self.qdrant_utils.chech_collection_exist("sample_collection")

        self.assertIsInstance(available, bool)
        self.assertFalse(available)

    def test_multiple_collections_available_given_input(self):
        """
        test multiple collections available with given input
        """
        collections = ["collection1", "collection2", "collection3"]
        for col in collections:
            self.qdrant_client.create_collection(
                col,
                vectors_config=models.VectorParams(
                    size=100, distance=models.Distance.COSINE
                ),
            )

        available = self.qdrant_utils.chech_collection_exist("collection1")

        self.assertIsInstance(available, bool)
        self.assertTrue(available)
