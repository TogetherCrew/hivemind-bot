from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine

from .dual_qdrant_retrieval_engine import DualQdrantRetrievalEngine


class BaseQdrantEngine:
    def __init__(self, platform_name: str, community_id: str) -> None:
        """
        initialize the qdrant db engine to query the database related to a community
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

    def prepare(self, testing=False) -> BaseQueryEngine:
        engine = DualQdrantRetrievalEngine.setup_engine(
            llm=Settings.llm,
            synthesizer=get_response_synthesizer(),
            platform_name=self.platform_name,
            community_id=self.community_id,
        )

        return engine
