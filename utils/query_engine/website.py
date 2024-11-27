from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class WebsiteQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str) -> None:
        platform_name = "website"
        super().__init__(platform_name, community_id)
