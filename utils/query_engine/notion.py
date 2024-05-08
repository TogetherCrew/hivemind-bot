from utils.query_engine.base_engine import BaseEngine


class NotionQueryEngine(BaseEngine):
    def __init__(self, community_id: str) -> None:
        platform_name = "notion"
        super().__init__(platform_name, community_id)
