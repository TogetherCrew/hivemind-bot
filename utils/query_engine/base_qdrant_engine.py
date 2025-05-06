from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine

from .dual_qdrant_retrieval_engine import DualQdrantRetrievalEngine


class BaseQdrantEngine:
    def __init__(self, platform_id: str, community_id: str) -> None:
        """
        initialize the qdrant db engine to query the database related to a community
        and the table related to the platform

        Parameters
        -----------
        platform_id : str
            the ID representing a specific platform
        community_id : str
            the database for a community
            normally the community database is saved as
            `community_{community_id}`
        """
        self.platform_id = platform_id
        self.community_id = community_id

    def prepare(
        self, enable_answer_skipping: bool = False, testing=False
    ) -> BaseQueryEngine:
        engine = DualQdrantRetrievalEngine.setup_engine(
            llm=Settings.llm,
            synthesizer=get_response_synthesizer(),
            platform_id=self.platform_id,
            community_id=self.community_id,
            enable_answer_skipping=enable_answer_skipping,
        )

        return engine
