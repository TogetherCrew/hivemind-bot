from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class NotionQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str) -> None:
        platform_name = "notion"
        super().__init__(platform_name, community_id)
