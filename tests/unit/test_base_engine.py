from unittest import TestCase

from utils.query_engine.base_engine import BaseEngine


class TestBaseEngine(TestCase):

    def test_setup_vector_store_index(self):
        """
        Tests that _setup_vector_store_index creates a PGVectorAccess object
        and calls its load_index method.
        """
        platform_table_name = "test_table"
        dbname = "test_db"

        base_engine = BaseEngine._setup_vector_store_index(
            platform_table_name=platform_table_name,
            dbname=dbname,
            testing=True,
        )
        self.assertIn(dbname, base_engine.vector_store.connection_string)
        self.assertEqual(base_engine.vector_store.table_name, platform_table_name)
