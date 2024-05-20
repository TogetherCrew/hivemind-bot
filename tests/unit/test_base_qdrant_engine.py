from unittest import TestCase

from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class TestBaseQdrantEngine(TestCase):
    def test_setup_vector_store_index(self):
        """
        Tests that _setup_vector_store_index creates a PGVectorAccess object
        and calls its load_index method.
        """
        platform_table_name = "test_table"
        community_id = "123456"
        base_engine = BaseQdrantEngine(
            platform_name=platform_table_name,
            community_id=community_id,
        )
        base_engine = base_engine._setup_vector_store_index(
            testing=True,
        )

        expected_collection_name = f"{community_id}_{platform_table_name}"
        self.assertEqual(
            base_engine.vector_store.collection_name, expected_collection_name
        )
