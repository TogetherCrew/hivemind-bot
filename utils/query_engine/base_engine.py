from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from bot.retrievers.utils.load_hyperparams import load_hyperparams
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from tc_hivemind_backend.qdrant_vector_access import QDrantVectorAccess


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
        self.collection_name = f"{self.community_id}_{platform_name}"

    def prepare(self, testing=False):
        vector_store_index = self._setup_vector_store_index(
            testing=testing,
        )
        _, similarity_top_k, _ = load_hyperparams()

        retriever = VectorIndexRetriever(
            index=vector_store_index,
            similarity_top_k=similarity_top_k,
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
            collection_name : str
                to override the default collection_name
        """
        collection_name = kwargs.get("collection_name", self.collection_name)
        qdrant_vector = QDrantVectorAccess(
            collection_name=collection_name,
            testing=testing,
        )
        index = qdrant_vector.load_index()
        return index
