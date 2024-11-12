from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from typing import List
from starlette.status import HTTP_403_FORBIDDEN


# List of valid API keys - in production, this should be stored securely
VALID_API_KEYS = ["key1", "key2", "test_key"]
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="No API key provided"
        )

    if api_key_header not in VALID_API_KEYS:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")

    return api_key_header
