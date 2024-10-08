import unittest
from unittest.mock import patch
import mongomock

from schema import PayloadModel
from utils.persist_payload import PersistPayload


class TestPersistPayloadIntegration(unittest.TestCase):
    """Integration tests for the PersistPayload class."""

    # Sample PayloadModel data
    sample_payload_data = {
        "communityId": "650be9f4e2c1234abcd12345",
        "route": {
            "source": "api-gateway",
            "destination": {"queue": "data-processing", "event": "new-data"},
        },
        "question": {
            "message": "What is the meaning of life?",
            "filters": {"category": "philosophy"},
        },
        "response": {
            "message": "The meaning of life is subjective and varies from person to person."
        },
        "metadata": {"timestamp": "2023-10-08T12:00:00"},
    }

    @patch("utils.mongo.MongoSingleton.get_instance")
    def setUp(self, mock_mongo_instance):
        """Setup a mocked MongoDB client for testing."""
        # Create a mock MongoDB client using `mongomock`
        self.mock_client = mongomock.MongoClient()

        # Mock the `get_client` method to return the mocked client
        mock_instance = mock_mongo_instance.return_value
        mock_instance.get_client.return_value = self.mock_client

        # Initialize the class under test with the mocked MongoDB client
        self.persist_payload = PersistPayload()

    def test_persist_valid_payload(self):
        """Test persisting a valid PayloadModel into the database."""
        # Create a PayloadModel instance from the sample data
        payload = PayloadModel(**self.sample_payload_data)

        # Call the `persist` method to store the payload in the mock database
        self.persist_payload.persist(payload)

        # Retrieve the persisted document from the mock database
        persisted_data = self.mock_client["hivemind"]["messages"].find_one(
            {"communityId": self.sample_payload_data["communityId"]}
        )

        # Check that the persisted document matches the original payload
        self.assertIsNotNone(persisted_data)
        self.assertEqual(
            persisted_data["communityId"], self.sample_payload_data["communityId"]
        )
        self.assertEqual(
            persisted_data["route"]["source"],
            self.sample_payload_data["route"]["source"],
        )
        self.assertEqual(
            persisted_data["question"]["message"],
            self.sample_payload_data["question"]["message"],
        )

    def test_persist_with_invalid_payload(self):
        """Test that attempting to persist an invalid payload raises an exception."""
        # Create an invalid PayloadModel by omitting required fields
        invalid_payload_data = {
            "communityId": self.sample_payload_data["communityId"],
            "route": {},  # Invalid as required fields are missing
            "question": {"message": ""},
            "response": {"message": ""},
            "metadata": None,
        }

        # Construct the PayloadModel (this will raise a validation error)
        with self.assertRaises(ValueError):
            PayloadModel(**invalid_payload_data)

    def test_persist_handles_mongo_exception(self):
        """Test that MongoDB exceptions are properly handled and logged."""
        # Create a valid PayloadModel instance
        payload = PayloadModel(**self.sample_payload_data)

        # Simulate a MongoDB exception during the insert operation
        with patch.object(
            self.mock_client["hivemind"]["messages"],
            "insert_one",
            side_effect=Exception("Database error"),
        ):
            with self.assertLogs(level="ERROR") as log:
                self.persist_payload.persist(payload)
                self.assertIn(
                    "Failed to persist payload to database for community", log.output[0]
                )
