from bson import ObjectId
from utils.mongo import MongoSingleton


class DataSourceSelector:
    def select_data_source(self, community_id: str) -> dict[str, str]:
        """
        Given a community id, find all its data sources selected for hivemind module

        Parameters
        -----------
        community_id : str
            id of a community

        Returns
        ----------
        data_sources : dict[str, str]
            a dictionary representing what data sources is selected
            for the given community, with platform names as keys and
            platform IDs as values
        """
        db_results = self._query_modules_db(community_id)
        data_sources = {data["name"]: str(data["platform"]) for data in db_results}
        return data_sources

    def _query_modules_db(self, community_id: str) -> list[dict]:
        client = MongoSingleton.get_instance().get_client()
        hivemind_module = client["Core"]["modules"].find_one(
            {
                "community": ObjectId(community_id),
                "name": "hivemind",
            },
            {
                "options.platforms.name": 1,
                "options.platforms.platform": 1,
            },
        )
        if hivemind_module is None:
            platforms = {}
        else:
            platforms = hivemind_module["options"]["platforms"]

        return platforms
