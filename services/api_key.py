from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
from utils.mongo import MongoSingleton

# List of valid API keys - in production, this should be stored securely
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Dependency function to validate API key

    Parameters
    -------------
    api_key_header : str
        the api key passed to the header

    Raises
    ------
    HTTPException
        If API key is missing or invalid

    Returns
    -------
    api_key_header : str
        The validated API key
    """
    validator = ValidateAPIKey()

    if not api_key_header:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="No API key provided"
        )

    valid = await validator.validate(api_key_header)
    if not valid:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return api_key_header


class ValidateAPIKey:
    def __init__(self) -> None:
        self.client = MongoSingleton.get_instance().get_client()
        self.db = "hivemind"
        self.tokens_collection = "tokens"

    async def validate(self, api_key: str) -> bool:
        """
        check if the api key is available in mongodb or not

        Parameters
        ------------
        api_key : str
            the provided key to check in db

        Returns
        ---------
        valid : bool
            if the key was available in mongo collection, then return True
            else, the token is not valid and return False
        """
        document = self.client[self.db][self.tokens_collection].find_one(
            {"token": api_key}
        )

        return True if document else False
