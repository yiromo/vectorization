from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv
import hashlib

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str
    MONGO_DATABASE_NAME: str

    IS_DEVELOPMENT: bool = False
    
    model_config = SettingsConfigDict(env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore

settings = get_settings()
