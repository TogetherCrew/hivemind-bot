from bot.retrievers.custom_retriever import CustomVectorStoreRetriever
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import get_response_synthesizer, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from tc_hivemind_backend.embeddings import CohereEmbedding
from tc_hivemind_backend.pg_vector_access import PGVectorAccess


class BaseEngine:
    def __init__(self, platform_name: str, community_id: str) -> None:
        """
        initialize the engine to query the database related to a community
        and the table related to the platform

        Parameters
        -----------
        platform_name : str
            the table representative of data for a specific platform
        community_id : str
            the database for a community
            normally the community database is saved as
            `community_{community_id}`
        """
        self.platform_name = platform_name
        self.community_id = community_id

    def prepare(self, testing=False):
        dbname = f"community_{self.community_id}"

        index = self._setup_vector_store_index(
            platform_table_name=self.platform_name,
            dbname=dbname,
            testing=testing,
        )
        _, similarity_top_k, _ = load_hyperparams()

        retriever = CustomVectorStoreRetriever(
            index=index, similarity_top_k=similarity_top_k
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(),
        )
        return query_engine

    def _setup_vector_store_index(
        cls,
        platform_table_name: str,
        dbname: str,
        testing: bool = False,
    ) -> VectorStoreIndex:
        """
        prepare the vector_store for querying data
        """
        cls.platform_name = platform_table_name

        pg_vector = PGVectorAccess(
            table_name=platform_table_name,
            dbname=dbname,
            testing=testing,
            embed_model=CohereEmbedding(),
        )
        index = pg_vector.load_index()
        return index
