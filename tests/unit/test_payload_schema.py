from unittest import TestCase

from pydantic import ValidationError
from schema.payload import AMQPPayload


class TestPayloadModel(TestCase):
    """Test suite for AMQPPayload and its nested models."""

    valid_community_id = "650be9f4e2c1234abcd12345"

    # Helper function to create a valid payload dictionary
    def get_valid_payload(self):
        return {
            "communityId": self.valid_community_id,
            "route": {
                "source": "some-source",
                "destination": {"queue": "some-queue", "event": "some-event"},
            },
            "question": {
                "message": "What is the best approach?",
                "filters": {"category": "science"},
            },
            "response": {"message": "The best approach is using scientific methods."},
            "metadata": {"timestamp": "2023-10-08T12:00:00"},
        }

    def test_valid_payload(self):
        """Test if a valid payload is correctly validated."""
        payload = self.get_valid_payload()
        validated_model = AMQPPayload(**payload)
        self.assertEqual(validated_model.communityId, payload["communityId"])
        self.assertEqual(validated_model.route.source, payload["route"]["source"])
        self.assertEqual(
            validated_model.route.destination.queue,
            payload["route"]["destination"]["queue"],
        )

    def test_missing_required_field(self):
        """Test if missing a required field raises a ValidationError."""
        payload = self.get_valid_payload()
        del payload["route"]  # Remove a required field
        with self.assertRaises(ValidationError):
            AMQPPayload(**payload)

    def test_none_as_optional_fields(self):
        """Test if setting optional fields as None is valid."""
        payload = self.get_valid_payload()
        payload["route"]["destination"] = None  # Set optional destination to None
        payload["question"]["filters"] = None  # Set optional filters to None
        payload["metadata"] = None  # Set optional metadata to None
        validated_model = AMQPPayload(**payload)
        self.assertIsNone(validated_model.route.destination)
        self.assertIsNone(validated_model.question.filters)
        self.assertIsNone(validated_model.metadata)

    def test_invalid_route(self):
        """Test if an invalid RouteModel within AMQPPayload raises a ValidationError."""
        payload = self.get_valid_payload()
        payload["route"]["source"] = None  # Invalid value for a required field
        with self.assertRaises(ValidationError):
            AMQPPayload(**payload)

    def test_empty_string_fields(self):
        """Test if fields with empty strings are allowed."""
        payload = self.get_valid_payload()
        payload["route"]["source"] = ""  # Set an empty string
        payload["question"]["message"] = ""  # Set an empty string
        validated_model = AMQPPayload(**payload)
        self.assertEqual(validated_model.route.source, "")
        self.assertEqual(validated_model.question.message, "")
