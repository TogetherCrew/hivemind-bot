from tc_hivemind_backend.db.qdrant import QdrantSingleton


class QDrantUtils:
    def __init__(self) -> None:
        self.qdrant_client = QdrantSingleton.get_instance().get_client()

    def chech_collection_exist(self, collection_name: str) -> bool:
        """
        check if the collection exist on qdrant database
        """
        pass
