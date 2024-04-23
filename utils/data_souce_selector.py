from .mongo import MongoSingleton
from bson import ObjectId


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
        platforms = list(map(lambda data: data["platform"]["name"], db_results))
        data_sources = dict.fromkeys(platforms, True)
        return data_sources

    def _query_modules_db(self, community_id: str) -> list[dict]:
        client = MongoSingleton.get_instance().get_client()

        pipeline = [
            {"$match": {"name": "hivemind", "communityId": ObjectId(community_id)}},
            {"$unwind": "$options.platforms"},
            {
                "$lookup": {
                    "from": "platforms",
                    "localField": "options.platforms.platformId",
                    "foreignField": "_id",
                    "as": "platform",
                }
            },
            {"$unwind": "$platform"},
            {
                "$project": {
                    "_id": 0,
                    "platform.name": 1,
                }
            },
        ]
        cursor = client["Core"]["modules"].aggregate(pipeline)

        data_sources = list(cursor)
        return data_sources
