import logging

from utils.mongo import MongoSingleton
from schema import PayloadModel


class PersistPayload:
    def __init__(self) -> None:
        # the place we would save data in mongo
        self.db = "hivemind"
        self.collection = "internal_messages"
        self.client = MongoSingleton.get_instance().get_client()

    def persist(self, payload: PayloadModel) -> None:
        """
        persist the payload within the database

        Parameters
        -----------
        payload : schema.PayloadModel
            the data payload to save on database
        """
        community_id = payload.communityId
        try:
            self.client[self.db][self.collection].insert_one(payload.model_dump())
            logging.info(
                f"Payload for community id: {community_id} persisted successfully!"
            )
        except Exception as exp:
            logging.error(
                f"Failed to persist payload to database for community: {community_id}!"
                f"Exception: {exp}"
            )
