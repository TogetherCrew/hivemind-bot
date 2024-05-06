from bson import ObjectId
from utils.mongo import MongoSingleton


class DataSourceSelector:
    def select_data_source(self, community_id: str) -> dict[str, bool]:
        """
        Given a community id, find all its data sources selected for hivemind module

        Parameters
        -----------
        community_id : str
            id of a community

        Returns
        ----------
        data_sources : dict[str, bool]
            a dictionary representing what data sources is selcted
            for the given community
        """
        db_results = self._query_modules_db(community_id)
        platforms = list(map(lambda data: data["name"], db_results))
        data_sources = dict.fromkeys(platforms, True)
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
            },
        )
        if hivemind_module is None:
            raise ValueError(
                f"No hivemind modules set for the given community id: {community_id}"
            )
        platforms = hivemind_module["options"]["platforms"]

        return platforms
