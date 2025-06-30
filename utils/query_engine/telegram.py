from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine
from schema.type import DataType
from utils.query_engine import DualQdrantRetrievalEngine
from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class TelegramQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str, platform_id: str = None) -> None:
        # If no platform_id provided, use default collection name for backward compatibility
        platform_id = platform_id or "telegram"
        super().__init__(platform_id, community_id)

    def prepare(
        self, enable_answer_skipping: bool = False, testing=False
    ) -> BaseQueryEngine:
        """
        Override the prepare method to add EXCLUDED_DATE_MARGIN filtering
        """
        # Create the engine with the filtered retriever
        engine = DualQdrantRetrievalEngine.setup_engine(
            llm=Settings.llm,
            synthesizer=get_response_synthesizer(),
            platform_id=self.platform_id,
            community_id=self.community_id,
            enable_answer_skipping=enable_answer_skipping,
            metadata_date_key="createdAt",
            metadata_date_format=DataType.FLOAT,
        )

        return engine


class TelegramDualQueryEngine:
    def __init__(self, community_id: str, platform_id: str = None) -> None:
        # If no platform_id provided, use default collection name for backward compatibility
        self.platform_id = platform_id or "telegram"
        self.community_id = community_id

    def prepare(self, enable_answer_skipping: bool) -> BaseQueryEngine:
        engine = DualQdrantRetrievalEngine.setup_engine_with_summaries(
            llm=Settings.llm,
            synthesizer=get_response_synthesizer(),
            platform_id=self.platform_id,
            community_id=self.community_id,
            metadata_date_key="createdAt",
            metadata_date_format=DataType.FLOAT,
            metadata_date_summary_key="date",
            metadata_date_summary_format=DataType.STRING,
            enable_answer_skipping=enable_answer_skipping,
        )
        return engine
