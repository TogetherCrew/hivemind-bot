from datetime import datetime
from unittest import TestCase

from bson import ObjectId
from utils.data_source_selector import DataSourceSelector
from utils.mongo import MongoSingleton


class TestDataSourceSelector(TestCase):
    def setUp(self) -> None:
        self.client = MongoSingleton.get_instance().get_client()
        self.community_id = "6579c364f1120850414e0dc4"
        self.client["Core"].drop_collection("modules")
        self.client["Core"].drop_collection("platforms")

    def test_no_community(self):
        """
        test if no community selected hivemind modeules
        """
        selector = DataSourceSelector()
        data_sources = selector.select_data_source(community_id=self.community_id)
        self.assertEqual(data_sources, {})

    def test_single_platform(self):
        platform_id = "6579c364f1120850414e0da1"
        self.client["Core"]["platforms"].insert_one(
            {
                "_id": ObjectId(platform_id),
                "name": "discord",
                "metadata": {
                    "name": "TEST",
                    "channels": ["1234", "4321"],
                    "roles": ["111", "222"],
                },
                "community": ObjectId(self.community_id),
                "disconnectedAt": None,
                "connectedAt": datetime(2023, 12, 1),
                "createdAt": datetime(2023, 12, 1),
                "updatedAt": datetime(2023, 12, 1),
            }
        )

        self.client["Core"]["modules"].insert_one(
            {
                "name": "hivemind",
                "communityId": ObjectId(self.community_id),
                "options": {
                    "platforms": [
                        {
                            "platformId": ObjectId(platform_id),
                            "fromDate": datetime(2024, 1, 1),
                            "options": {},
                        }
                    ]
                },
            }
        )
        selector = DataSourceSelector()
        data_sources = selector.select_data_source(community_id=self.community_id)
        self.assertEqual(
            data_sources,
            {
                "discord": True,
            },
        )

    def test_multiple_platform(self):
        platform_id1 = "6579c364f1120850414e0da1"
        platform_id2 = "6579c364f1120850414e0da2"
        platform_id3 = "6579c364f1120850414e0da3"
        self.client["Core"]["platforms"].insert_many(
            [
                {
                    "_id": ObjectId(platform_id1),
                    "name": "discord",
                    "metadata": {
                        "name": "TEST",
                        "channels": ["1234", "4321"],
                        "roles": ["111", "222"],
                    },
                    "community": ObjectId(self.community_id),
                    "disconnectedAt": None,
                    "connectedAt": datetime(2023, 12, 1),
                    "createdAt": datetime(2023, 12, 1),
                    "updatedAt": datetime(2023, 12, 1),
                },
                {
                    "_id": ObjectId(platform_id2),
                    "name": "github",
                    "metadata": {
                        "organizationId": 12345,
                    },
                    "community": ObjectId(self.community_id),
                    "disconnectedAt": None,
                    "connectedAt": datetime(2023, 12, 1),
                    "createdAt": datetime(2023, 12, 1),
                    "updatedAt": datetime(2023, 12, 1),
                },
                {
                    "_id": ObjectId(platform_id3),
                    "name": "discourse",
                    "metadata": {
                        "some_id": 133445,
                    },
                    "community": ObjectId(self.community_id),
                    "disconnectedAt": None,
                    "connectedAt": datetime(2023, 12, 1),
                    "createdAt": datetime(2023, 12, 1),
                    "updatedAt": datetime(2023, 12, 1),
                },
            ]
        )

        self.client["Core"]["modules"].insert_one(
            {
                "name": "hivemind",
                "communityId": ObjectId(self.community_id),
                "options": {
                    "platforms": [
                        {
                            "platformId": ObjectId(platform_id1),
                            "fromDate": datetime(2024, 1, 1),
                            "options": {},
                        },
                        {
                            "platformId": ObjectId(platform_id2),
                            "fromDate": datetime(2024, 1, 1),
                            "options": {},
                        },
                        {
                            "platformId": ObjectId(platform_id3),
                            "fromDate": datetime(2024, 1, 1),
                            "options": {},
                        },
                    ]
                },
            }
        )
        selector = DataSourceSelector()
        data_sources = selector.select_data_source(community_id=self.community_id)
        self.assertEqual(
            data_sources,
            {
                "discord": True,
                "github": True,
                "discourse": True,
            },
        )
