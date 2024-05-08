from utils.query_engine.base_engine import BaseEngine


class GDriveQueryEngine(BaseEngine):
    def __init__(self, community_id: str) -> None:
        platform_name = "gdrive"
        super().__init__(platform_name, community_id)
