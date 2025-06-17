import logging
from datetime import datetime, timezone
from bson import ObjectId

from schema import HTTPPayload, RouteModelPayload
from utils.mongo import MongoSingleton


class PersistPayload:
    def __init__(self) -> None:
        # the place we would save data in mongo
        self.db = "hivemind"
        self.internal_msgs_collection = "internal_messages"
        self.external_msgs_collection = "external_messages"
        self.client = MongoSingleton.get_instance().get_client()

    def persist_payload(self, payload: RouteModelPayload, workflow_id: str | None = None) -> None:
        """
        persist the whole payload within the database

        Parameters
        -----------
        payload : schema.RouteModelPayload
            the data payload to save on database
        workflow_id : str | None
            if provided, update the existing document with this workflow_id
            if None, insert a new document
        """
        community_id = payload.communityId
        try:
            if workflow_id is None:
                # Insert new document (current behavior)
                self.client[self.db][self.internal_msgs_collection].insert_one(
                    {
                        **payload.model_dump(),
                        "createdAt": datetime.now().replace(tzinfo=timezone.utc),
                        "updatedAt": datetime.now().replace(tzinfo=timezone.utc),
                    }
                )
                logging.info(
                    f"New payload for community id: {community_id} persisted successfully!"
                )
            else:
                # Update existing document with workflow_id
                # Check if createdAt needs to be set if it doesn't exist
                self.client[self.db][self.internal_msgs_collection].update_one(
                    {"_id": ObjectId(workflow_id), "createdAt": {"$exists": False}},
                    {
                        "$set": {
                            "createdAt": datetime.now().replace(tzinfo=timezone.utc)
                        }
                    },
                )

                # Get existing document to merge metadata
                existing_doc = self.client[self.db][self.internal_msgs_collection].find_one(
                    {"_id": ObjectId(workflow_id)}
                )
                
                # Merge metadata if existing document has metadata
                merged_metadata = payload.metadata
                if existing_doc and "metadata" in existing_doc and existing_doc["metadata"]:
                    if merged_metadata is None:
                        merged_metadata = {}
                    # Merge existing metadata with new metadata (new metadata takes precedence)
                    merged_metadata = {**existing_doc["metadata"], **merged_metadata}

                # Prepare the update document
                if existing_doc:
                    # Update existing document - only update specific fields
                    update_doc = {
                        "$set": {
                            "metadata": merged_metadata,
                            "response": payload.response.model_dump(),
                            "updatedAt": datetime.now().replace(tzinfo=timezone.utc),
                        }
                    }
                else:
                    # Upsert new document - include all payload fields
                    update_doc = {
                        "$set": {
                            **payload.model_dump(),
                            "metadata": merged_metadata,
                            "updatedAt": datetime.now().replace(tzinfo=timezone.utc),
                        }
                    }

                # Update or upsert the main document with evaluation results and response
                self.client[self.db][self.internal_msgs_collection].update_one(
                    {"_id": ObjectId(workflow_id)},
                    update_doc,
                    upsert=True,
                )
                logging.info(
                    f"Updated payload for community id: {community_id} with workflow_id: {workflow_id} persisted successfully!"
                )
        except Exception as exp:
            logging.error(
                f"Failed to persist payload to database for community: {community_id}!"
                f"Exception: {exp}"
            )

    def persist_http(self, payload: HTTPPayload, update: bool = False) -> None:
        """
        persist the http payload in database

        Parameters
        -----------
        payload : schema.HTTPPayload
            the data payload to save on database
        update : bool
            to update the previous document matching task id
            default is set to False meaning just to add
        """
        community_id = payload.communityId
        try:
            if not update:
                self.client[self.db][self.external_msgs_collection].insert_one(
                    {
                        **payload.model_dump(),
                        "createdAt": datetime.now().replace(tzinfo=timezone.utc),
                        "updatedAt": datetime.now().replace(tzinfo=timezone.utc),
                    }
                )
                logging.info(
                    "Added HTTP Payload for community id: "
                    f"{community_id} persisted successfully!"
                )
            else:
                # Check if createdAt needs to be set if it doesn't exist
                self.client[self.db][self.external_msgs_collection].update_one(
                    {"taskId": payload.taskId, "createdAt": {"$exists": False}},
                    {
                        "$set": {
                            "createdAt": datetime.now().replace(tzinfo=timezone.utc)
                        }
                    },
                )

                # Update or upsert the main document
                self.client[self.db][self.external_msgs_collection].update_one(
                    {"taskId": payload.taskId},
                    {
                        "$set": {
                            **payload.model_dump(),
                            "updatedAt": datetime.now().replace(tzinfo=timezone.utc),
                        }
                    },
                    upsert=True,
                )
                logging.info(
                    "Upserted HTTP Payload for community id: "
                    f"{community_id} persisted successfully!"
                )
        except Exception as exp:
            logging.error(
                f"Failed to persist payload to database for community: {community_id}!"
                f"Exception: {exp}"
            )
