from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine
from schema.type import DataType
from utils.query_engine import DualQdrantRetrievalEngine
from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class GitHubQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str) -> None:
        platform_name = "github"
        super().__init__(platform_name, community_id)


class GitHubDualQueryEngine:
    def __init__(self, community_id: str) -> None:
        self.platform_name = "github"
        self.community_id = community_id

    def prepare(self) -> BaseQueryEngine:
        engine = DualQdrantRetrievalEngine.setup_engine_with_summaries(
            llm=Settings.llm,
            synthesizer=get_response_synthesizer(),
            platform_name=self.platform_name,
            community_id=self.community_id,
            metadata_date_key="created_at",
            metadata_date_format=DataType.FLOAT,
            metadata_date_summary_key="date",
            metadata_date_summary_format=DataType.FLOAT,
        )
        return engine
