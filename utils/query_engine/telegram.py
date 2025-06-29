from datetime import datetime, timedelta, timezone

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine
from qdrant_client.http import models
from schema.type import DataType
from utils.globals import EXCLUDED_DATE_MARGIN
from utils.query_engine import DualQdrantRetrievalEngine
from utils.query_engine.base_qdrant_engine import BaseQdrantEngine
from utils.query_engine.qa_prompt import qa_prompt
from bot.retrievers.utils.load_hyperparams import load_hyperparams


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
        # Setup the vector store index same as the parent class
        collection_name = f"{self.community_id}_{self.platform_id}"
        vector_store_index = DualQdrantRetrievalEngine._setup_vector_store_index(
            collection_name=collection_name
        )

        # Create the date filter for EXCLUDED_DATE_MARGIN
        latest_query_date = (
            datetime.now(tz=timezone.utc) - timedelta(minutes=EXCLUDED_DATE_MARGIN)
        ).timestamp()

        # For Telegram, we assume the metadata date key is "createdAt" and format is FLOAT
        # This matches what's used in TelegramDualQueryEngine
        must_filters = [
            models.FieldCondition(
                key="createdAt",
                range=models.Range(
                    lte=latest_query_date,
                ),
            )
        ]

        filter = models.Filter(must=must_filters)

        # Create a custom retriever with the date filter
        _, raw_data_top_k, date_margin = load_hyperparams()
        
        retriever = vector_store_index.as_retriever(
            vector_store_kwargs={"qdrant_filters": filter},
            similarity_top_k=raw_data_top_k,
        )

        # Set the required class attributes before creating the instance (same as setup_engine method)
        DualQdrantRetrievalEngine._date_margin = date_margin
        DualQdrantRetrievalEngine._enable_answer_skipping = enable_answer_skipping
        DualQdrantRetrievalEngine.summary_retriever = None

        # Create the engine with the filtered retriever
        engine = DualQdrantRetrievalEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(),
            llm=Settings.llm,
            qa_prompt=qa_prompt,
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
