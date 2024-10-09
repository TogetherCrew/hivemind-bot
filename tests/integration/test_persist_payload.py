import unittest
from unittest.mock import patch
import mongomock

from schema import AMQPPayload, QuestionModel, ResponseModel, HTTPPayload
from utils.persist_payload import PersistPayload


class TestPersistPayloadIntegration(unittest.TestCase):
    """Integration tests for the PersistPayload class."""

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

        # Sample AMQPPayload data
        self.sample_payload_data = {
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

        # Sample HTTPPayload data
        self.sample_http_payload_data = {
            "communityId": "650be9f4e2c1234abcd12345",
            "taskId": "task-123",
            "question": {
                "message": "What is the HTTP response code?",
                "filters": {"type": "http"},
            },
            "response": {"message": "OK"},
        }

        # Define a separate collection name for HTTP payloads in the PersistPayload class
        self.persist_payload = PersistPayload()
        self.persist_payload.external_msgs_collection = "http_messages"

    def test_persist_valid_payload(self):
        """Test persisting a valid AMQPPayload into the database."""
        # Create a AMQPPayload instance from the sample data
        payload = AMQPPayload(**self.sample_payload_data)

        # Call the `persist_amqp` method to store the payload in the mock database
        self.persist_payload.persist_amqp(payload)

        # Retrieve the persisted document from the mock database
        persisted_data = self.mock_client["hivemind"]["internal_messages"].find_one(
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
        # Create an invalid AMQPPayload by omitting required fields
        invalid_payload_data = {
            "communityId": self.sample_payload_data["communityId"],
            "route": {},  # Invalid as required fields are missing
            "question": {"message": ""},
            "response": {"message": ""},
            "metadata": None,
        }

        # Construct the AMQPPayload (this will raise a validation error)
        with self.assertRaises(ValueError):
            AMQPPayload(**invalid_payload_data)

    def test_persist_handles_mongo_exception(self):
        """Test that MongoDB exceptions are properly handled and logged."""
        # Create a valid AMQPPayload instance
        payload = AMQPPayload(**self.sample_payload_data)

        # Simulate a MongoDB exception during the insert operation
        with patch.object(
            self.mock_client["hivemind"]["internal_messages"],
            "insert_one",
            side_effect=Exception("Database error"),
        ):
            with self.assertLogs(level="ERROR") as log:
                self.persist_payload.persist_amqp(payload)
                print("log.output", log.output)
                self.assertIn(
                    "Failed to persist payload to database for community", log.output[0]
                )

    def test_persist_http_insert(self):
        """Test inserting a new HTTPPayload into the database."""
        # Create an HTTPPayload instance from the sample data
        question_model = QuestionModel(**self.sample_http_payload_data["question"])
        response_model = ResponseModel(**self.sample_http_payload_data["response"])
        http_payload = HTTPPayload(
            communityId=self.sample_http_payload_data["communityId"],
            taskId=self.sample_http_payload_data["taskId"],
            question=question_model,
            response=response_model,
        )

        # Call the `persist_http` method to store the payload in the mock database
        self.persist_payload.persist_http(http_payload)

        # Retrieve the persisted document from the mock database
        persisted_data = self.mock_client["hivemind"]["http_messages"].find_one(
            {"communityId": self.sample_http_payload_data["communityId"]}
        )

        # Check that the persisted document matches the original payload
        self.assertIsNotNone(persisted_data)
        self.assertEqual(
            persisted_data["communityId"], self.sample_http_payload_data["communityId"]
        )
        self.assertEqual(
            persisted_data["taskId"], self.sample_http_payload_data["taskId"]
        )
        self.assertEqual(
            persisted_data["response"]["message"],
            self.sample_http_payload_data["response"]["message"],
        )

    def test_persist_http_update(self):
        """Test updating an existing HTTPPayload document in the database."""
        # Insert an initial HTTP payload document into the mock database
        initial_data = self.sample_http_payload_data.copy()
        initial_data["response"]["message"] = "Not Found"  # Initial message
        self.mock_client["hivemind"]["http_messages"].insert_one(initial_data)

        # Create an updated HTTPPayload instance
        question_model = QuestionModel(**self.sample_http_payload_data["question"])
        updated_response_model = ResponseModel(message="OK")  # Updated response message
        updated_http_payload = HTTPPayload(
            communityId=self.sample_http_payload_data["communityId"],
            taskId=self.sample_http_payload_data["taskId"],
            question=question_model,
            response=updated_response_model,
        )

        # Call the `persist_http` method to update the existing document
        self.persist_payload.persist_http(updated_http_payload, update=True)

        # Retrieve the updated document from the mock database
        updated_data = self.mock_client["hivemind"]["http_messages"].find_one(
            {"taskId": self.sample_http_payload_data["taskId"]}
        )

        # Check that the document was updated correctly
        self.assertIsNotNone(updated_data)
        self.assertEqual(
            updated_data["taskId"], self.sample_http_payload_data["taskId"]
        )
        self.assertEqual(updated_data["response"]["message"], "OK")  # Verify update

    def test_persist_http_upsert(self):
        """Test upsert behavior when updating a non-existent document."""
        # Create an HTTPPayload instance from the sample data
        question_model = QuestionModel(**self.sample_http_payload_data["question"])
        response_model = ResponseModel(**self.sample_http_payload_data["response"])
        http_payload = HTTPPayload(
            communityId=self.sample_http_payload_data["communityId"],
            taskId=self.sample_http_payload_data["taskId"],
            question=question_model,
            response=response_model,
        )

        # Ensure the document does not exist before upsert
        initial_check = self.mock_client["hivemind"]["http_messages"].find_one(
            {"taskId": self.sample_http_payload_data["taskId"]}
        )
        self.assertIsNone(initial_check)

        # Call the `persist_http` method with update=True to perform upsert
        self.persist_payload.persist_http(http_payload, update=True)

        # Check that the document now exists in the collection
        upserted_data = self.mock_client["hivemind"]["http_messages"].find_one(
            {"taskId": self.sample_http_payload_data["taskId"]}
        )
        self.assertIsNotNone(upserted_data)
        self.assertEqual(
            upserted_data["taskId"], self.sample_http_payload_data["taskId"]
        )

    def test_persist_http_handles_mongo_exception(self):
        """Test that MongoDB exceptions are properly handled and logged."""
        # Create a valid HTTPPayload instance
        question_model = QuestionModel(**self.sample_http_payload_data["question"])
        response_model = ResponseModel(**self.sample_http_payload_data["response"])
        http_payload = HTTPPayload(
            communityId=self.sample_http_payload_data["communityId"],
            taskId=self.sample_http_payload_data["taskId"],
            question=question_model,
            response=response_model,
        )

        # Simulate a MongoDB exception during the insert operation
        with patch.object(
            self.mock_client["hivemind"]["http_messages"],
            "insert_one",
            side_effect=Exception("Database error"),
        ):
            with self.assertLogs(level="ERROR") as log:
                self.persist_payload.persist_http(http_payload)
                self.assertIn(
                    "Failed to persist payload to database for community", log.output[0]
                )

        # Simulate a MongoDB exception during the update operation
        with patch.object(
            self.mock_client["hivemind"]["http_messages"],
            "update_one",
            side_effect=Exception("Database update error"),
        ):
            with self.assertLogs(level="ERROR") as log:
                self.persist_payload.persist_http(http_payload, update=True)
                self.assertIn(
                    "Failed to persist payload to database for community", log.output[0]
                )
