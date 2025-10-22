"""
Classification API endpoints.

This module defines all endpoints related to content classification,
including single review classification and batch processing.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, status

from app.schemas.classification import (
    ReviewRequest,
    ClassificationResponse,
    BatchClassificationResponse,
    BatchReviewItem,
    ToxicityResult,
    SpamResult,
)
from app.services.classifier import ClassifierService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post(
    "",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a single review",
    description="""
    Classify a music album review for toxic content and spam.

    This endpoint analyzes the provided text using two ML models:
    - **Toxicity Detection**: Identifies offensive language and hate speech
    - **Spam Detection**: Detects promotional content and spam

    The service returns detailed scores for both classifications and provides
    a recommendation on whether the content should be removed.

    **Thresholds**:
    - Toxicity threshold: {toxic_threshold}
    - Spam threshold: {spam_threshold}

    Content is flagged for removal if either score exceeds its threshold.
    """.format(
        toxic_threshold=settings.toxic_threshold,
        spam_threshold=settings.spam_threshold,
    ),
    responses={
        200: {
            "description": "Successful classification",
            "content": {
                "application/json": {
                    "example": {
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
                }
            },
        },
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {"detail": "The text cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error in classification: model not loaded"}
                }
            },
        },
    },
)
async def classify_review(review: ReviewRequest) -> ClassificationResponse:
    """
    Classify a single review for toxic content and spam.

    Args:
        review: ReviewRequest containing the text to classify

    Returns:
        ClassificationResponse with results from both models and recommendation

    Raises:
        HTTPException: If classification fails
    """
    logger.info(f"Classifying review: {review.text[:50]}...")

    try:
        # Classify using both models
        toxicity_result = ClassifierService.classify_toxicity(review.text)
        spam_result = ClassifierService.classify_spam(review.text)

        # Generate recommendation
        recommendation, should_remove = ClassifierService.generate_recommendation(
            toxicity_result, spam_result
        )

        return ClassificationResponse(
            toxicity=ToxicityResult(**toxicity_result),
            spam=SpamResult(**spam_result),
            recommendation=recommendation,
            should_be_removed=should_remove,
        )
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in classification: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify multiple reviews in batch",
    description=f"""
    Classify multiple music album reviews in a single request.

    This endpoint is useful for processing reported reviews in bulk.
    Each review is classified independently, and results are returned
    for all reviews even if some fail.

    **Limits**:
    - Maximum reviews per batch: {settings.max_batch_size}

    Reviews that fail classification will have an error field instead
    of classification results.
    """,
    responses={
        200: {
            "description": "Batch classification completed",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "text": "Excelente álbum",
                                "classification": {
                                    "toxicity": {
                                        "label": "NON_TOXIC",
                                        "score_toxic": 0.0156,
                                        "score_non_toxic": 0.9844,
                                        "confidence": 0.9844,
                                    },
                                    "spam": {
                                        "label": "NOT_SPAM",
                                        "score_spam": 0.0423,
                                        "score_not_spam": 0.9577,
                                        "confidence": 0.9577,
                                    },
                                    "recommendation": "KEEP",
                                    "should_be_removed": False,
                                },
                                "error": None,
                            }
                        ],
                        "total": 1,
                        "successful": 1,
                        "failed": 0,
                    }
                }
            },
        },
        400: {
            "description": "Invalid batch size",
            "content": {
                "application/json": {
                    "example": {
                        "detail": f"Maximum {settings.max_batch_size} reviews per batch"
                    }
                }
            },
        },
    },
)
async def classify_batch(reviews: List[ReviewRequest]) -> BatchClassificationResponse:
    """
    Classify multiple reviews in a single batch request.

    Args:
        reviews: List of ReviewRequest objects to classify

    Returns:
        BatchClassificationResponse containing results for all reviews

    Raises:
        HTTPException: If batch size exceeds maximum
    """
    if len(reviews) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {settings.max_batch_size} reviews per batch",
        )

    logger.info(f"Processing batch of {len(reviews)} reviews")

    results = []
    successful = 0
    failed = 0

    for review in reviews:
        try:
            # Classify using both models
            toxicity_result = ClassifierService.classify_toxicity(review.text)
            spam_result = ClassifierService.classify_spam(review.text)

            # Generate recommendation
            recommendation, should_remove = ClassifierService.generate_recommendation(
                toxicity_result, spam_result
            )

            classification = ClassificationResponse(
                toxicity=ToxicityResult(**toxicity_result),
                spam=SpamResult(**spam_result),
                recommendation=recommendation,
                should_be_removed=should_remove,
            )

            results.append(
                BatchReviewItem(
                    text=review.text, classification=classification, error=None
                )
            )
            successful += 1
        except Exception as e:
            logger.error(f"Error classifying review '{review.text[:50]}': {str(e)}")
            results.append(
                BatchReviewItem(text=review.text, classification=None, error=str(e))
            )
            failed += 1

    return BatchClassificationResponse(
        results=results, total=len(reviews), successful=successful, failed=failed
    )
