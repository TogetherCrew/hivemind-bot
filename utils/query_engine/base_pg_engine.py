from bot.retrievers.custom_retriever import CustomVectorStoreRetriever
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from tc_hivemind_backend.embeddings import CohereEmbedding
from tc_hivemind_backend.pg_vector_access import PGVectorAccess


class BasePGEngine:
    def __init__(self, platform_name: str, community_id: str) -> None:
        """
        initialize the pg vector db engine to query the database related to a community
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
        self.dbname = f"community_{self.community_id}"

    def prepare(self, testing=False):
        index = self._setup_vector_store_index(
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
        self,
        testing: bool = False,
        **kwargs,
    ) -> VectorStoreIndex:
        """
        prepare the vector_store for querying data

        Parameters
        ------------
        testing : bool
            for testing purposes
        **kwargs :
            table_name : str
                to override the default table_name
            dbname : str
                to override the default database name
        """
        table_name = kwargs.get("table_name", self.platform_name)
        dbname = kwargs.get("dbname", self.dbname)

        embed_model: BaseEmbedding
        if testing:
            from llama_index.core import MockEmbedding

            embed_model = MockEmbedding(embed_dim=1024)
        else:
            embed_model = CohereEmbedding()

        pg_vector = PGVectorAccess(
            table_name=table_name,
            dbname=dbname,
            testing=testing,
            embed_model=embed_model,
        )
        index = pg_vector.load_index()
        return index