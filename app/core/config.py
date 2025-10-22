"""
Application configuration and settings.

This module contains all configuration constants and settings for the application.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings.

    Attributes:
        app_name: Name of the application
        app_version: Current version
        app_description: Short description of the service
        host: Host address to bind to
        port: Port to bind to
        toxic_threshold: Minimum confidence score to classify as toxic (0.0-1.0)
        spam_threshold: Minimum confidence score to classify as spam (0.0-1.0)
        max_batch_size: Maximum number of reviews that can be processed in a batch
        max_text_length: Maximum length of text to process (in tokens)
        toxicity_model: Hugging Face model ID for toxicity detection
        spam_model: Hugging Face model ID for spam detection
    """

    # Application metadata
    app_name: str = "Content Moderation ML Service"
    app_version: str = "1.0.0"
    app_description: str = (
        "Machine learning microservice for classifying toxic content and spam "
        "in Spanish music album reviews"
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Model thresholds
    toxic_threshold: float = 0.7
    spam_threshold: float = 0.7

    # Processing limits
    max_batch_size: int = 50
    max_text_length: int = 512

    # ML Models
    toxicity_model: str = "bgonzalezbustamante/bert-spanish-toxicity"
    spam_model: str = "asfilcnx3/spam-detection-es"

    # Environment
    environment: str = "production"
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
