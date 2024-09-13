import os

from dotenv import load_dotenv


def load_postgres_credentials() -> dict[str, str]:
    """
    load postgresql db credentials from .env

    Returns:
    ---------
    postgres_creds : dict[str, Any]
        postgresql credentials
        a dictionary representative of
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
        a dictionary representative of
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


def load_mongo_credentials() -> dict[str, str]:
    """
    load mongo db credentials from .env

    Returns:
    ---------
    mongo_creds : dict[str, Any]
        mongodb credentials
        a dictionary representative of
            `user`: str
            `password` : str
            `host` : str
            `port` : int
    """
    load_dotenv()

    mongo_creds = {}

    mongo_creds["user"] = os.getenv("MONGODB_USER", "")
    mongo_creds["password"] = os.getenv("MONGODB_PASS", "")
    mongo_creds["host"] = os.getenv("MONGODB_HOST", "")
    mongo_creds["port"] = os.getenv("MONGODB_PORT", "")

    return mongo_creds

def load_redis_credentials() -> dict[str, str]:
    """
    load redis db credentials from .env

    Returns:
    ---------
    redis_creds : dict[str, Any]
        redis credentials
        a dictionary representative of
            `user`: str
            `password` : str
            `host` : str
            `port` : int
            `url` : str
    """
    load_dotenv()
    
    redis_creds = {}

    user = os.getenv("REDIS_USER", "")
    password = os.getenv("REDIS_PASSWORD", "")
    host = os.getenv("REDIS_HOST", "")
    port = os.getenv("REDIS_PORT", "")

    redis_creds["user"] = user
    redis_creds["password"] = password
    redis_creds["host"] = host
    redis_creds["port"] = port
    redis_creds["url"] = f"redis://{user}:{password}@{host}:{port}"

    return redis_creds