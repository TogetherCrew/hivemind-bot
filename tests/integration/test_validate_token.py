from unittest import IsolatedAsyncioTestCase

from services.api_key import ValidateAPIKey
from utils.mongo import MongoSingleton


class TestValidateToken(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        """
        Set up test case with a test database
        """
        self.client = MongoSingleton.get_instance().get_client()
        self.validator = ValidateAPIKey()

        # Using test database to avoid affecting production data
        self.validator.db = "hivemind_test"
        self.validator.tokens_collection = "tokens_test"

        # Clean start for each test
        self.clean_database()

    async def asyncTearDown(self) -> None:
        """
        Clean up test database after each test
        """
        self.clean_database()

    def clean_database(self) -> None:
        """
        Helper method to clean the test database
        """
        self.client.drop_database(self.validator.db)

    async def test_no_token_available(self):
        """
        Test validation when no tokens exist in database
        """
        api_key = "1234"
        community = await self.validator.validate(api_key)

        self.assertIsNone(community)

    async def test_no_matching_token_available(self):
        """
        Test validation when tokens exist but none match
        """
        # Insert test tokens - no await needed as this is synchronous
        self.client[self.validator.db][self.validator.tokens_collection].insert_many(
            [
                {
                    "id": 1,
                    "token": "1111",
                    "community": "AAAA",
                    "options": {},
                },
                {
                    "id": 2,
                    "token": "2222",
                    "community": "BBBB",
                    "options": {},
                },
                {
                    "id": 3,
                    "token": "3333",
                    "community": "CCCC",
                    "options": {},
                },
            ]
        )

        api_key = "1234"
        community = await self.validator.validate(api_key)

        self.assertIsNone(community)

    async def test_single_token_available(self):
        """
        Test validation when matching token exists
        """
        api_key = "1234"

        self.client[self.validator.db][self.validator.tokens_collection].insert_many(
            [
                {
                    "id": 1,
                    "token": api_key,
                    "community": "AAAA",
                    "options": {},
                },
                {
                    "id": 2,
                    "token": "2222",
                    "community": "BBBB",
                    "options": {},
                },
                {
                    "id": 3,
                    "token": "3333",
                    "community": "CCCC",
                    "options": {},
                },
            ]
        )

        community = await self.validator.validate(api_key)

        self.assertIsNotNone(community)
        self.assertEqual(community, "AAAA")

    async def test_validation_with_empty_api_key(self):
        """
        Test validation with empty API key
        """
        community = await self.validator.validate("")

        self.assertIsNone(community)
