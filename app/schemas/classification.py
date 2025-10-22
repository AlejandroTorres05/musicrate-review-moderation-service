"""
Pydantic schemas for classification requests and responses.

This module defines all the data models used for API request validation
and response serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class RecommendationType(str, Enum):
    """Possible recommendation types for content moderation."""

    KEEP = "KEEP"
    REMOVE_TOXIC = "REMOVE_TOXIC"
    REMOVE_SPAM = "REMOVE_SPAM"
    REMOVE_BOTH = "REMOVE_BOTH"


class ReviewRequest(BaseModel):
    """
    Request model for review classification.

    Attributes:
        text: The review text to classify (in Spanish)
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The review text to analyze",
        examples=["Me encanta este álbum, las letras son increíbles"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """Validate that text is not empty or only whitespace."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Este álbum es una obra maestra, las melodías son increíbles"
                },
                {
                    "text": "Odio este álbum, el artista es un idiota"
                },
                {
                    "text": "¡Compra ahora! Descuentos en nuestra tienda"
                },
            ]
        }
    }


class ToxicityResult(BaseModel):
    """
    Toxicity classification result.

    Attributes:
        label: Classification label (TOXIC or NON_TOXIC)
        score_toxic: Probability that the content is toxic (0.0-1.0)
        score_non_toxic: Probability that the content is non-toxic (0.0-1.0)
        confidence: Highest probability score
    """

    label: str = Field(
        ...,
        description="Classification label",
        examples=["TOXIC", "NON_TOXIC"],
    )
    score_toxic: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of toxic content",
        examples=[0.8923],
    )
    score_non_toxic: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of non-toxic content",
        examples=[0.1077],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (highest probability)",
        examples=[0.8923],
    )


class SpamResult(BaseModel):
    """
    Spam classification result.

    Attributes:
        label: Classification label (SPAM or NOT_SPAM)
        score_spam: Probability that the content is spam (0.0-1.0)
        score_not_spam: Probability that the content is not spam (0.0-1.0)
        confidence: Highest probability score
    """

    label: str = Field(
        ...,
        description="Classification label",
        examples=["SPAM", "NOT_SPAM"],
    )
    score_spam: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of spam content",
        examples=[0.9123],
    )
    score_not_spam: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of legitimate content",
        examples=[0.0877],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (highest probability)",
        examples=[0.9123],
    )


class ClassificationResponse(BaseModel):
    """
    Complete classification response for a single review.

    Attributes:
        toxicity: Toxicity classification results
        spam: Spam classification results
        recommendation: Action recommendation (KEEP, REMOVE_TOXIC, REMOVE_SPAM, REMOVE_BOTH)
        should_be_removed: Whether the content should be removed based on thresholds
    """

    toxicity: ToxicityResult = Field(
        ...,
        description="Toxicity classification results",
    )
    spam: SpamResult = Field(
        ...,
        description="Spam classification results",
    )
    recommendation: RecommendationType = Field(
        ...,
        description="Moderation action recommendation",
    )
    should_be_removed: bool = Field(
        ...,
        description="Whether the content should be removed",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "toxicity": {
                        "label": "NON_TOXIC",
                        "score_toxic": 0.0234,
                        "score_non_toxic": 0.9766,
                        "confidence": 0.9766,
                    },
                    "spam": {
                        "label": "NOT_SPAM",
                        "score_spam": 0.0512,
                        "score_not_spam": 0.9488,
                        "confidence": 0.9488,
                    },
                    "recommendation": "KEEP",
                    "should_be_removed": False,
                }
            ]
        }
    }


class BatchReviewItem(BaseModel):
    """Single item in a batch classification response."""

    text: str = Field(..., description="The original review text")
    classification: Optional[ClassificationResponse] = Field(
        None,
        description="Classification results if successful",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if classification failed",
    )


class BatchClassificationResponse(BaseModel):
    """
    Response for batch classification requests.

    Attributes:
        results: List of classification results for each review
        total: Total number of reviews processed
        successful: Number of successful classifications
        failed: Number of failed classifications
    """

    results: List[BatchReviewItem] = Field(
        ...,
        description="Classification results for each review",
    )
    total: int = Field(
        ...,
        description="Total number of reviews processed",
    )
    successful: int = Field(
        ...,
        description="Number of successful classifications",
    )
    failed: int = Field(
        ...,
        description="Number of failed classifications",
    )


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status: Overall service status
        toxicity_model_loaded: Whether toxicity model is loaded
        spam_model_loaded: Whether spam model is loaded
        device: Device being used (cpu or cuda)
        version: API version
    """

    status: str = Field(
        ...,
        description="Overall service status",
        examples=["healthy"],
    )
    toxicity_model_loaded: bool = Field(
        ...,
        description="Whether toxicity detection model is loaded",
    )
    spam_model_loaded: bool = Field(
        ...,
        description="Whether spam detection model is loaded",
    )
    device: str = Field(
        ...,
        description="Computation device (cpu or cuda)",
        examples=["cuda:0", "cpu"],
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0"],
    )


class ErrorResponse(BaseModel):
    """
    Error response model.

    Attributes:
        detail: Error message detail
        error_code: Optional error code
    """

    detail: str = Field(
        ...,
        description="Error message",
        examples=["The text cannot be empty"],
    )
    error_code: Optional[str] = Field(
        None,
        description="Error code for programmatic handling",
        examples=["EMPTY_TEXT", "MODEL_ERROR"],
    )
