"""
Machine learning model loading and management.

This module handles loading and managing the ML models used for
toxicity and spam detection.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)


class MLModels:
    """
    Singleton class for managing ML models.

    This class handles loading and storing the toxicity and spam detection models,
    ensuring they are only loaded once and reused throughout the application lifecycle.

    Attributes:
        toxicity_tokenizer: Tokenizer for toxicity detection model
        toxicity_model: BERT model for toxicity detection
        spam_tokenizer: Tokenizer for spam detection model
        spam_model: Model for spam detection
        device: PyTorch device (cuda or cpu)
    """

    _instance: Optional["MLModels"] = None
    _initialized: bool = False

    def __new__(cls) -> "MLModels":
        """Ensure only one instance of MLModels exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(MLModels, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the ML models (only once)."""
        if not MLModels._initialized:
            self.toxicity_tokenizer: Optional[AutoTokenizer] = None
            self.toxicity_model: Optional[AutoModelForSequenceClassification] = None
            self.spam_tokenizer: Optional[AutoTokenizer] = None
            self.spam_model: Optional[AutoModelForSequenceClassification] = None
            self.device: Optional[torch.device] = None
            MLModels._initialized = True

    async def load_models(self) -> None:
        """
        Load both toxicity and spam detection models.

        This method loads the models from Hugging Face and moves them to the
        appropriate device (GPU if available, otherwise CPU).

        Raises:
            Exception: If models fail to load
        """
        logger.info("Starting to load ML models...")

        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            # Load toxicity detection model
            logger.info("Loading toxicity detection model...")
            self.toxicity_tokenizer = AutoTokenizer.from_pretrained(
                settings.toxicity_model
            )
            self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                settings.toxicity_model
            )
            self.toxicity_model.to(self.device)
            self.toxicity_model.eval()
            logger.info("Toxicity model loaded successfully")

            # Load spam detection model
            logger.info("Loading spam detection model...")
            self.spam_tokenizer = AutoTokenizer.from_pretrained(settings.spam_model)
            self.spam_model = AutoModelForSequenceClassification.from_pretrained(
                settings.spam_model
            )
            self.spam_model.to(self.device)
            self.spam_model.eval()
            logger.info("Spam model loaded successfully")

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def models_loaded(self) -> bool:
        """
        Check if all models are loaded.

        Returns:
            True if all models are loaded, False otherwise
        """
        return (
            self.toxicity_model is not None
            and self.toxicity_tokenizer is not None
            and self.spam_model is not None
            and self.spam_tokenizer is not None
        )

    def get_toxicity_model(
        self,
    ) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Get the toxicity detection model and tokenizer.

        Returns:
            Tuple of (tokenizer, model)

        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.toxicity_model or not self.toxicity_tokenizer:
            raise RuntimeError("Toxicity model not loaded")
        return self.toxicity_tokenizer, self.toxicity_model

    def get_spam_model(
        self,
    ) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Get the spam detection model and tokenizer.

        Returns:
            Tuple of (tokenizer, model)

        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.spam_model or not self.spam_tokenizer:
            raise RuntimeError("Spam model not loaded")
        return self.spam_tokenizer, self.spam_model


# Global instance
ml_models = MLModels()
