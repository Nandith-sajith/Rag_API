from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str  # Add this

    class Config:
        env_file = ".env"

settings = Settings()