from tc_hivemind_backend.db.qdrant import QdrantSingleton


class QDrantUtils:
    def __init__(self, community_id: str) -> None:
        """
        setup qdrant utils for a specific community

        Parameters
        ------------
        community_id : str
            the community we want to initialize the utils for

        """
        self.qdrant_client = QdrantSingleton.get_instance().get_client()
        self.community_id = community_id

    def chech_collection_exist(self, platform_name: str) -> bool:
        """
        check if the collection exist on qdrant database

        Parameters
        -----------
        platform_name : str
            the platform name we want to check for its collection availability

        Returns
        --------
        available : bool
            if the collection was available True, else would be False
        """
        collection_name = f"{self.community_id}_{platform_name}"
        available = self.qdrant_client.collection_exists(collection_name)
        return available
