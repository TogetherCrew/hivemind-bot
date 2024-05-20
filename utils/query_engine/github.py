from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class GitHubQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str) -> None:
        platform_name = "github"
        super().__init__(platform_name, community_id)
