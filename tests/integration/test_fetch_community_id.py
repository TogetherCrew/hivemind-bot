from datetime import datetime, timedelta
from unittest import TestCase

from bson import ObjectId
from utils.fetch_community_id import fetch_community_id_by_guild_id
from utils.mongo import MongoSingleton


class TestFetchDiscordCommunityId(TestCase):
    def add_platform(self):
        client = MongoSingleton.get_instance().get_client()

        action = {
            "INT_THR": 1,
            "UW_DEG_THR": 1,
            "PAUSED_T_THR": 1,
            "CON_T_THR": 4,
            "CON_O_THR": 3,
            "EDGE_STR_THR": 5,
            "UW_THR_DEG_THR": 5,
            "VITAL_T_THR": 4,
            "VITAL_O_THR": 3,
            "STILL_T_THR": 2,
            "STILL_O_THR": 2,
            "DROP_H_THR": 2,
            "DROP_I_THR": 1,
        }

        client["Core"]["platforms"].insert_one(
            {
                "_id": ObjectId(self.platform_id),
                "name": "discord",
                "metadata": {
                    "id": self.guild_id,
                    "icon": "111111111111111111111111",
                    "name": "A guild",
                    "selectedChannels": [
                        {"channelId": "1020707129214111827", "channelName": "general"}
                    ],
                    "window": {"period_size": 7, "step_size": 1},
                    "action": action,
                    "period": datetime.now() - timedelta(days=30),
                },
                "community": ObjectId(self.community_id),
                "disconnectedAt": None,
                "connectedAt": (datetime.now() - timedelta(days=40)),
                "isInProgress": True,
                "createdAt": datetime(2023, 11, 1),
                "updatedAt": datetime(2023, 11, 1),
            }
        )

    def delete_platform(self):
        client = MongoSingleton.get_instance().get_client()
        client["Core"]["platforms"].delete_one({"_id": ObjectId(self.platform_id)})

    def test_get_guild_id(self):
        self.platform_id = "515151515151515151515151"
        self.guild_id = "1234"
        self.community_id = "aabbccddeeff001122334455"
        self.delete_platform()
        self.add_platform()

        community_id = fetch_community_id_by_guild_id(guild_id=self.guild_id)

        self.assertEqual(community_id, self.community_id)

    def test_get_guild_id_no_data(self):
        self.platform_id = "515151515151515151515151"
        self.guild_id = "1234"
        self.community_id = "aabbccddeeff001122334455"

        self.delete_platform()
        self.add_platform()

        client = MongoSingleton.get_instance().get_client()
        client["Core"]["platforms"].delete_one({"_id": ObjectId(self.platform_id)})

        with self.assertRaises(ValueError):
            _ = fetch_community_id_by_guild_id(guild_id=self.guild_id)
