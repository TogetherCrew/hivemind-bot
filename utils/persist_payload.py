import logging
from datetime import datetime, timezone

from schema import RouteModelPayload, HTTPPayload
from utils.mongo import MongoSingleton


class PersistPayload:
    def __init__(self) -> None:
        # the place we would save data in mongo
        self.db = "hivemind"
        self.internal_msgs_collection = "internal_messages"
        self.external_msgs_collection = "external_messages"
        self.client = MongoSingleton.get_instance().get_client()

    def persist_payload(self, payload: RouteModelPayload) -> None:
        """
        persist the whole payload within the database

        Parameters
        -----------
        payload : schema.RouteModelPayload
            the data payload to save on database
        """
        community_id = payload.communityId
        try:
            self.client[self.db][self.internal_msgs_collection].insert_one(
                {
                    **payload.model_dump(),
                    "createdAt": datetime.now().replace(tzinfo=timezone.utc),
                    "updatedAt": datetime.now().replace(tzinfo=timezone.utc),
                }
            )
            logging.info(
                f"Payload for community id: {community_id} persisted successfully!"
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
