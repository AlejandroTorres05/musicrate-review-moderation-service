"""
Classification service for toxicity and spam detection.

This module contains the business logic for classifying text content
using the loaded ML models.
"""

import logging
import torch
from typing import Dict

from app.models.ml_models import ml_models
from app.core.config import settings
from app.schemas.classification import RecommendationType

logger = logging.getLogger(__name__)


class ClassifierService:
    """
    Service for classifying text content.

    This service provides methods for detecting toxicity and spam in text,
    and generating moderation recommendations based on the classification results.
    """

    @staticmethod
    def classify_toxicity(text: str) -> Dict[str, float]:
        """
        Classify text for toxic content.

        Args:
            text: The text to classify

        Returns:
            Dictionary containing:
                - label: "TOXIC" or "NON_TOXIC"
                - score_toxic: Probability of toxic content (0.0-1.0)
                - score_non_toxic: Probability of non-toxic content (0.0-1.0)
                - confidence: Highest probability score

        Raises:
            RuntimeError: If models are not loaded
            Exception: If classification fails
        """
        try:
            tokenizer, model = ml_models.get_toxicity_model()

            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=settings.max_text_length,
            )
            inputs = {k: v.to(ml_models.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)

            # Convert logits to probabilities
            probs = torch.softmax(outputs.logits, dim=-1)
            score_toxic = float(probs[0][1].item())
            score_non_toxic = float(probs[0][0].item())

            label = "TOXIC" if score_toxic > score_non_toxic else "NON_TOXIC"

            return {
                "label": label,
                "score_toxic": round(score_toxic, 4),
                "score_non_toxic": round(score_non_toxic, 4),
                "confidence": round(max(score_toxic, score_non_toxic), 4),
            }
        except Exception as e:
            logger.error(f"Error in toxicity classification: {str(e)}")
            raise

    @staticmethod
    def classify_spam(text: str) -> Dict[str, float]:
        """
        Classify text for spam content.

        Args:
            text: The text to classify

        Returns:
            Dictionary containing:
                - label: "SPAM" or "NOT_SPAM"
                - score_spam: Probability of spam content (0.0-1.0)
                - score_not_spam: Probability of legitimate content (0.0-1.0)
                - confidence: Highest probability score

        Raises:
            RuntimeError: If models are not loaded
            Exception: If classification fails
        """
        try:
            tokenizer, model = ml_models.get_spam_model()

            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=settings.max_text_length,
            )
            inputs = {k: v.to(ml_models.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)

            # Convert logits to probabilities
            probs = torch.softmax(outputs.logits, dim=-1)
            spam_score = float(probs[0][1].item())
            not_spam_score = float(probs[0][0].item())

            label = "SPAM" if spam_score > not_spam_score else "NOT_SPAM"

            return {
                "label": label,
                "score_spam": round(spam_score, 4),
                "score_not_spam": round(not_spam_score, 4),
                "confidence": round(max(spam_score, not_spam_score), 4),
            }
        except Exception as e:
            logger.error(f"Error in spam classification: {str(e)}")
            raise

    @staticmethod
    def generate_recommendation(
        toxicity_result: Dict[str, float], spam_result: Dict[str, float]
    ) -> tuple[RecommendationType, bool]:
        """
        Generate a moderation recommendation based on classification results.

        Args:
            toxicity_result: Results from toxicity classification
            spam_result: Results from spam classification

        Returns:
            Tuple of (recommendation, should_be_removed):
                - recommendation: RecommendationType enum value
                - should_be_removed: Boolean indicating if content should be removed
        """
        is_toxic = toxicity_result["label"] == "TOXIC"
        is_spam = spam_result["label"] == "SPAM"

        # Check if scores exceed thresholds
        should_remove = (
            is_toxic and toxicity_result["score_toxic"] >= settings.toxic_threshold
        ) or (is_spam and spam_result["score_spam"] >= settings.spam_threshold)

        # Determine recommendation
        if should_remove:
            if is_toxic and is_spam:
                recommendation = RecommendationType.REMOVE_BOTH
            elif is_toxic:
                recommendation = RecommendationType.REMOVE_TOXIC
            else:
                recommendation = RecommendationType.REMOVE_SPAM
        else:
            recommendation = RecommendationType.KEEP

        return recommendation, should_remove
