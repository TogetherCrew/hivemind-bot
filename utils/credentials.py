import os

from dotenv import load_dotenv


def load_postgres_credentials() -> dict[str, str]:
    """
    load posgresql db credentials from .env

    Returns:
    ---------
    postgres_creds : dict[str, Any]
        postgresql credentials
        a dictionary representive of
            `user`: str
            `password` : str
            `host` : str
            `port` : int
    """
    load_dotenv()

    postgres_creds = {}

    postgres_creds["user"] = os.getenv("POSTGRES_USER", "")
    postgres_creds["password"] = os.getenv("POSTGRES_PASS", "")
    postgres_creds["host"] = os.getenv("POSTGRES_HOST", "")
    postgres_creds["port"] = os.getenv("POSTGRES_PORT", "")

    return postgres_creds


def load_rabbitmq_credentials() -> dict[str, str]:
    """
    load rabbitmq credentials from .env

    Returns:
    ---------
    rabbitmq_creds : dict[str, Any]
        rabbitmq credentials
        a dictionary representive of
            `user`: str
            `password` : str
            `host` : str
            `port` : int
    """
    load_dotenv()

    rabbitmq_creds = {}

    rabbitmq_creds["user"] = os.getenv("RABBIT_USER", "")
    rabbitmq_creds["password"] = os.getenv("RABBIT_PASSWORD", "")
    rabbitmq_creds["host"] = os.getenv("RABBIT_HOST", "")
    rabbitmq_creds["port"] = os.getenv("RABBIT_PORT", "")

    return rabbitmq_creds
