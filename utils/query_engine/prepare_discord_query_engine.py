from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine
from schema.type import DataType
from utils.query_engine import DualQdrantRetrievalEngine
from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class DiscordQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str, platform_id: str | None = None) -> None:
        # If no platform_id provided, use default collection name for backward compatibility
        platform_id = platform_id or "discord"
        super().__init__(platform_id, community_id)

    def prepare(
        self, enable_answer_skipping: bool = False, testing=False
    ) -> BaseQueryEngine:
        """
        Override the prepare method to add EXCLUDED_DATE_MARGIN filtering
        """
        # Create the engine with the filtered retriever
        engine = DualQdrantRetrievalEngine.setup_engine(
            llm=Settings.llm,  # type: ignore
            synthesizer=get_response_synthesizer(),
            platform_id=self.platform_id,
            community_id=self.community_id,
            enable_answer_skipping=enable_answer_skipping,
            metadata_date_key="date",
            metadata_date_format=DataType.FLOAT,
        )

        return engine


class DiscordDualQueryEngine:
    def __init__(self, community_id: str, platform_id: str | None = None) -> None:
        # If no platform_id provided, use default collection name for backward compatibility
        self.platform_id = platform_id or "discord"
        self.community_id = community_id

    def prepare(self, enable_answer_skipping: bool) -> BaseQueryEngine:
        engine = DualQdrantRetrievalEngine.setup_engine_with_summaries(
            llm=Settings.llm,  # type: ignore
            synthesizer=get_response_synthesizer(),
            platform_id=self.platform_id,
            community_id=self.community_id,
            metadata_date_key="date",
            metadata_date_format=DataType.FLOAT,
            metadata_date_summary_key="date",
            metadata_date_summary_format=DataType.FLOAT,
            enable_answer_skipping=enable_answer_skipping,
            summary_type="day",
        )
        return engine


def prepare_discord_engine(
    community_id: str,
    platform_id: str,
    enable_answer_skipping: bool,
) -> BaseQueryEngine:
    """
    query the platform database using filters given
    and give an anwer to the given query using the LLM

    Parameters
    ------------
    community_id : str
        the discord community id data to query
    platform_id : str
        the platform id to query


    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """
    # Use the new DiscordQueryEngine instead of LevelBasedPlatformQueryEngine
    engine = DiscordQueryEngine(community_id=community_id, platform_id=platform_id)
    return engine.prepare(enable_answer_skipping=enable_answer_skipping)


def prepare_discord_engine_auto_filter(
    community_id: str,
    platform_id: str,
    enable_answer_skipping: bool,
) -> BaseQueryEngine:
    """
    get the query engine and do the filtering automatically.
    By automatically we mean, it would first query the summaries
    to get the metadata filters

    Parameters
    -----------
    community_id : str
        the discord community data to query
    platform_id : str
        the platform id to query


    Returns
    ---------
    query_engine : BaseQueryEngine
        the created query engine with the filters
    """
    # Use the new DiscordDualQueryEngine for auto filtering with summaries
    engine = DiscordDualQueryEngine(community_id=community_id, platform_id=platform_id)
    return engine.prepare(enable_answer_skipping=enable_answer_skipping)
