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

    def test_no_community(self):
        """
        test if no community selected hivemind modeules
        """
        selector = DataSourceSelector()
        data_sources = selector.select_data_source(community_id=self.community_id)
        self.assertEqual(data_sources, {})

    def test_single_platform(self):
        platform_id = "6579c364f1120850414e0da1"
        self.client["Core"]["modules"].insert_one(
            {
                "name": "hivemind",
                "community": ObjectId(self.community_id),
                "options": {
                    "platforms": [
                        {
                            "platform": ObjectId(platform_id),
                            "name": "discord",
                            "metadata": {
                                "fromDate": datetime(2024, 1, 1),
                            },
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

        self.client["Core"]["modules"].insert_one(
            {
                "name": "hivemind",
                "community": ObjectId(self.community_id),
                "options": {
                    "platforms": [
                        {
                            "platform": ObjectId(platform_id1),
                            "name": "discord",
                            "metadata": {
                                "fromDate": datetime(2024, 1, 1),
                            },
                        },
                        {
                            "platform": ObjectId(platform_id2),
                            "name": "github",
                            "metadata": {
                                "fromDate": datetime(2024, 1, 1),
                            },
                        },
                        {
                            "platform": ObjectId(platform_id3),
                            "name": "discourse",
                            "metadata": {
                                "fromDate": datetime(2024, 1, 1),
                            },
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
