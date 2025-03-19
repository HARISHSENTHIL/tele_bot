import os
from functools import lru_cache
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN: str
    WEBHOOK_URL: str
    WEBHOOK_SECRET: str
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    
    # Image Generation Settings
    MAX_CONCURRENT_GENERATION: int = 5
    GENERATION_TIMEOUT: int = 60
    
    # OpenAI Settings (for prompt enhancement)
    OPENAI_API_KEY: str = ""
    
    # FLUX Model Settings
    LORA_ADAPTER_PATH: str = "./flux-lora-adapters-00/flux-lora-adapters-00.safetensors"
    USE_CUDA: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached settings instance to avoid reloading from environment
    every time settings are accessed.
    """
    return Settings() 