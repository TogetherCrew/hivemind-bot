from llama_index.core import VectorStoreIndex
from tc_hivemind_backend.embeddings import CohereEmbedding
from tc_hivemind_backend.pg_vector_access import PGVectorAccess


class BaseEngine:
    @classmethod
    def _setup_vector_store_index(
        cls,
        platform_table_name: str,
        dbname: str,
        testing: bool = False,
    ) -> VectorStoreIndex:
        """
        prepare the vector_store for querying data
        """
        pg_vector = PGVectorAccess(
            table_name=platform_table_name,
            dbname=dbname,
            testing=testing,
            embed_model=CohereEmbedding(),
        )
        index = pg_vector.load_index()
        return index
