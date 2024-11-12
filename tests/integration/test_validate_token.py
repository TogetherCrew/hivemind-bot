from unittest import TestCase

from services.api_key import ValidateAPIKey
from utils.mongo import MongoSingleton


class TestValidateToken(TestCase):
    def setUp(self) -> None:
        self.client = MongoSingleton.get_instance().get_client()
        self.validator = ValidateAPIKey()

        # changing the db so not to overlap with the right ones
        self.validator.db = "hivemind_test"
        self.validator.tokens_collection = "tokens_test"

        self.client.drop_database(self.validator.db)

    def tearDown(self) -> None:
        self.client.drop_database(self.validator.db)

    def test_no_token_available(self):
        api_key = "1234"
        valid = self.validator.validate(api_key)

        self.assertEqual(valid, False)

    def test_no_matching_token_available(self):
        self.client[self.validator.db][self.validator.tokens_collection].insert_many(
            [
                {
                    "id": 1,
                    "token": "1111",
                    "options": {},
                },
                {
                    "id": 2,
                    "token": "2222",
                    "options": {},
                },
                {
                    "id": 3,
                    "token": "3333",
                    "options": {},
                },
            ]
        )
        api_key = "1234"
        valid = self.validator.validate(api_key)

        self.assertEqual(valid, False)

    def test_single_token_available(self):
        api_key = "1234"
        self.client[self.validator.db][self.validator.tokens_collection].insert_many(
            [
                {
                    "id": 1,
                    "token": api_key,
                    "options": {},
                },
                {
                    "id": 2,
                    "token": "2222",
                    "options": {},
                },
                {
                    "id": 3,
                    "token": "3333",
                    "options": {},
                },
            ]
        )
        valid = self.validator.validate(api_key)

        self.assertEqual(valid, True)
