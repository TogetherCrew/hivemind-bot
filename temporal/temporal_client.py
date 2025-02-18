import os

from dotenv import load_dotenv
from temporalio.client import Client


class TemporalClient:
    def __init__(self) -> None:
        load_dotenv()

    async def get_client(self) -> Client:
        credentials = self._load_credentials()

        url: str = credentials["host"] + ":" + credentials["port"]
        client = await Client.connect(url, api_key=credentials["api_key"])

        return client

    def _load_credentials(self) -> dict[str, str]:
        """
        load the credentials for temporal

        Returns
        ------------
        credentials : dict[str, str]
            a dictionary holding temporal credentials
            {
                'host': str,
                'api_key': str,
                'port': str
            }
        """
        host = os.getenv("TEMPORAL_HOST")
        api_key = os.getenv("TEMPORAL_API_KEY")
        port = os.getenv("TEMPORAL_PORT")

        if not host:
            raise ValueError(
                "`TEMPORAL_HOST` is not configured right in env credentials!"
            )
        if not port:
            raise ValueError(
                "`TEMPORAL_PORT` is not configured right in env credentials!"
            )
        if api_key is None:
            raise ValueError(
                "`TEMPORAL_API_KEY` is not configured right in env credentials!"
            )

        credentials: dict[str, str] = {
            "host": host,
            "api_key": api_key,
            "port": port,
        }

        return credentials
