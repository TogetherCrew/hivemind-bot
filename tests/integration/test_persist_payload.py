import unittest
from unittest.mock import patch
import copy

import mongomock
from bson import ObjectId
from schema import HTTPPayload, QuestionModel, ResponseModel, RouteModelPayload
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

        # Sample RouteModelPayload data
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
        """Test persisting a valid RouteModelPayload into the database."""
        # Create a RouteModelPayload instance from the sample data
        payload = RouteModelPayload(**self.sample_payload_data)

        # Call the `persist_payload` method to store the payload in the mock database
        self.persist_payload.persist_payload(payload)

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
        # Create an invalid RouteModelPayload by omitting required fields
        invalid_payload_data = {
            "communityId": self.sample_payload_data["communityId"],
            "route": {},  # Invalid as required fields are missing
            "question": {"message": ""},
            "response": {"message": ""},
            "metadata": None,
        }

        # Construct the RouteModelPayload (this will raise a validation error)
        with self.assertRaises(ValueError):
            RouteModelPayload(**invalid_payload_data)

    def test_persist_handles_mongo_exception(self):
        """Test that MongoDB exceptions are properly handled and logged."""
        # Create a valid RouteModelPayload instance
        payload = RouteModelPayload(**self.sample_payload_data)

        # Simulate a MongoDB exception during the insert operation
        with patch.object(
            self.mock_client["hivemind"]["internal_messages"],
            "insert_one",
            side_effect=Exception("Database error"),
        ):
            with self.assertLogs(level="ERROR") as log:
                self.persist_payload.persist_payload(payload)
                print("log.output", log.output)
                self.assertIn(
                    "Failed to persist payload to database for community", log.output[0]
                )

    def test_persist_payload_with_workflow_id_update(self):
        """Test updating an existing document when workflow_id is provided."""
        # Create a RouteModelPayload instance from the sample data
        payload = RouteModelPayload(**self.sample_payload_data)
        workflow_id = "507f1f77bcf86cd799439011"  # Valid ObjectId format

        # First, insert an initial document with the workflow_id and existing metadata
        initial_data = copy.deepcopy(self.sample_payload_data)
        initial_data["_id"] = ObjectId(workflow_id)
        initial_data["response"]["message"] = "Initial response"
        initial_data["metadata"] = {
            "existing_key": "existing_value",
            "timestamp": "2023-10-08T12:00:00",
        }
        self.mock_client["hivemind"]["internal_messages"].insert_one(initial_data)

        # Update the payload with new response and new metadata
        updated_payload = RouteModelPayload(**self.sample_payload_data)
        updated_payload.response.message = "Updated response"
        updated_payload.metadata = {
            "new_key": "new_value",
            "answer_relevance_score": 0.95,
        }

        # Call the `persist_payload` method with workflow_id to update the existing document
        self.persist_payload.persist_payload(updated_payload, workflow_id=workflow_id)

        # Retrieve the updated document from the mock database
        updated_data = self.mock_client["hivemind"]["internal_messages"].find_one(
            {"_id": ObjectId(workflow_id)}
        )

        # Check that the document was updated correctly
        self.assertIsNotNone(updated_data)
        self.assertEqual(updated_data["_id"], ObjectId(workflow_id))
        self.assertEqual(updated_data["response"]["message"], "Updated response")
        self.assertEqual(
            updated_data["communityId"], self.sample_payload_data["communityId"]
        )

        # Check that metadata was merged correctly (existing + new metadata)
        expected_metadata = {
            "existing_key": "existing_value",
            "timestamp": "2023-10-08T12:00:00",
            "new_key": "new_value",
            "answer_relevance_score": 0.95,
        }
        self.assertEqual(updated_data["metadata"], expected_metadata)

    def test_persist_payload_with_workflow_id_upsert(self):
        """Test upsert behavior when workflow_id is provided but document doesn't exist."""
        # Create a RouteModelPayload instance from the sample data
        payload = RouteModelPayload(**self.sample_payload_data)
        workflow_id = "507f1f77bcf86cd799439012"  # Valid ObjectId format

        # Ensure the document does not exist before upsert
        initial_check = self.mock_client["hivemind"]["internal_messages"].find_one(
            {"_id": ObjectId(workflow_id)}
        )
        self.assertIsNone(initial_check)

        # Call the `persist_payload` method with workflow_id to perform upsert
        self.persist_payload.persist_payload(payload, workflow_id=workflow_id)

        # Check that the document now exists in the collection
        upserted_data = self.mock_client["hivemind"]["internal_messages"].find_one(
            {"_id": ObjectId(workflow_id)}
        )
        self.assertIsNotNone(upserted_data)
        self.assertEqual(upserted_data["_id"], ObjectId(workflow_id))
        self.assertEqual(
            upserted_data["communityId"], self.sample_payload_data["communityId"]
        )
        self.assertEqual(
            upserted_data["response"]["message"],
            self.sample_payload_data["response"]["message"],
        )

    def test_persist_payload_without_workflow_id_insert(self):
        """Test inserting new document when workflow_id is None (default behavior)."""
        # Create a RouteModelPayload instance from the sample data
        payload = RouteModelPayload(**self.sample_payload_data)

        # Call the `persist_payload` method without workflow_id (default behavior)
        self.persist_payload.persist_payload(payload, workflow_id=None)

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
        # Ensure workflow_id is not present in the document
        self.assertNotIn("workflow_id", persisted_data)

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
        initial_data = copy.deepcopy(self.sample_http_payload_data)
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
