from utils.query_engine.base_qdrant_engine import BaseQdrantEngine


class GDriveQueryEngine(BaseQdrantEngine):
    def __init__(self, community_id: str, platform_id: str = None) -> None:
        # If no platform_id provided, use default collection name for backward compatibility
        platform_id = platform_id or "google"
        super().__init__(platform_id, community_id)
