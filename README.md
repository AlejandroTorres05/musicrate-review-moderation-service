# Content Moderation ML Microservice

A FastAPI-based machine learning microservice for content moderation of music album reviews. This service uses transformer models to detect toxic content and spam in Spanish text, providing automated content filtering for reported reviews.

## Overview

This microservice is part of a larger music album review platform. When users report reviews for inappropriate content, this ML service analyzes the reported text to determine if it should be removed based on:

- **Toxicity Detection**: Identifies hate speech, offensive language, and toxic content
- **Spam Detection**: Detects spam, promotional content, and irrelevant messages

The service provides classification scores and actionable recommendations to help moderators make informed decisions.

## Quick Start

### Using Docker (Recommended)

```bash
# Start the service
docker-compose up -d

# Test the API
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Este álbum es increíble"}'

# View documentation
# Open http://localhost:8000/docs in your browser
```

### Using Python Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Project Structure

This project follows FastAPI best practices with a modular architecture:

```
ml-microservice/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application entry point
│   ├── api/                 # API layer
│   │   ├── __init__.py
│   │   └── routes/          # API route handlers
│   │       ├── __init__.py
│   │       ├── classification.py  # Classification endpoints
│   │       └── health.py          # Health check endpoints
│   ├── core/                # Core application configuration
│   │   ├── __init__.py
│   │   └── config.py        # Settings and configuration
│   ├── models/              # ML model management
│   │   ├── __init__.py
│   │   └── ml_models.py     # Model loading and inference
│   ├── schemas/             # Pydantic models (request/response)
│   │   ├── __init__.py
│   │   └── classification.py  # Classification schemas
│   └── services/            # Business logic
│       ├── __init__.py
│       └── classifier.py    # Classification service
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Docker Compose orchestration
├── requirements.txt         # Python dependencies
├── .dockerignore           # Docker build exclusions
├── .env.example            # Environment variables template
├── test_api.sh             # API testing script
└── README.md               # This file
```

### Architecture Overview

- **`app/main.py`**: Application entry point with FastAPI instance, middleware, and router registration
- **`app/api/routes/`**: API endpoints organized by feature (classification, health checks)
- **`app/core/config.py`**: Centralized configuration using Pydantic Settings
- **`app/models/ml_models.py`**: Singleton pattern for ML model management
- **`app/schemas/`**: Pydantic models for request validation and response serialization
- **`app/services/`**: Business logic separated from API routes

## Technical Architecture

### ML Models

The service uses two pre-trained transformer models from Hugging Face:

1. **Toxicity Classifier**: `bgonzalezbustamante/bert-spanish-toxicity`

   - BERT-based model fine-tuned for Spanish toxic content detection
   - Classifies text as TOXIC or NON_TOXIC
   - Provides confidence scores for both labels

2. **Spam Classifier**: `asfilcnx3/spam-detection-es`
   - Spanish spam detection model
   - Classifies text as SPAM or NOT_SPAM
   - Optimized for detecting promotional and irrelevant content

### How It Works

1. **Model Loading**: On startup, both models are loaded into memory and moved to the appropriate device (GPU if available, CPU otherwise)

2. **Text Processing**: When a review is submitted:

   - Text is tokenized using model-specific tokenizers
   - Tokens are truncated to 512 max length
   - Inputs are padded to ensure consistent tensor sizes

3. **Classification**: Each model:

   - Processes the tokenized input through BERT layers
   - Generates logits for each class
   - Applies softmax to convert logits to probabilities
   - Returns confidence scores for each label

4. **Decision Making**:
   - Scores are compared against configurable thresholds (default: 0.7)
   - A recommendation is generated based on classification results
   - The service indicates whether the content should be removed

### API Endpoints

#### Health & Monitoring

- `GET /` - Root endpoint with basic service information
- `GET /health` - Detailed health status with model information
- `GET /readiness` - Kubernetes-style readiness probe
- `GET /liveness` - Kubernetes-style liveness probe

#### Classification

- `POST /classify` - Classify a single review for toxicity and spam
- `POST /classify/batch` - Classify multiple reviews in batch (max 50)

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

### Linux Installation

1. **Clone the repository**

```bash
cd ml-microservice
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
```

3. **Activate the virtual environment**

```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the service**

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Windows Installation

1. **Clone the repository**

```cmd
cd ml-microservice
```

2. **Create a virtual environment**

```cmd
python3 -m venv venv
```

3. **Activate the virtual environment**

```cmd
venv\Scripts\activate
```

4. **Install dependencies**

```cmd
pip install -r requirements.txt
```

5. **Run the service**

```cmd
REM Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

REM Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Installation

#### Using Docker Compose (Recommended)

The easiest way to run the service locally with Docker:

1. **Make sure Docker and Docker Compose are installed**

   - Linux: Follow [Docker Engine installation](https://docs.docker.com/engine/install/)
   - Windows: Install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

2. **Start the service**

```bash
# Build and start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

3. **Access the service**

   - API: `http://localhost:8000`
   - Swagger docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

4. **Stop the service**

```bash
# Stop containers
docker-compose down

# Stop and remove volumes (this will delete cached models)
docker-compose down -v
```

#### Using Docker directly

If you prefer to use Docker without docker-compose:

```bash
# Build the image
docker build -t ml-content-moderation .

# Run the container
docker run -d \
  --name ml-microservice \
  -p 8000:8000 \
  -v ml-models-cache:/app/.cache/huggingface \
  ml-content-moderation

# View logs
docker logs -f ml-microservice

# Stop the container
docker stop ml-microservice

# Remove the container
docker rm ml-microservice
```

#### Development Mode with Docker

For development with hot-reload:

1. **Uncomment the volume mount in docker-compose.yml**:

```yaml
volumes:
  - ./app:/app/app # Uncomment this line
```

2. **Run with development command**:

```bash
docker-compose run --rm --service-ports ml-microservice \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Benefits of Using Docker

- **Consistent Environment**: Same setup across different operating systems
- **Easy Deployment**: Ready for production deployment
- **Model Caching**: Hugging Face models are cached in a volume, so they don't need to be downloaded on every restart
- **Resource Limits**: Configure CPU and memory limits in docker-compose.yml
- **Isolated Dependencies**: No conflicts with other Python projects on your system

## Usage

### API Documentation

Once the service is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Request Examples

#### Single Review Classification

**Request:**

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Este álbum es una obra maestra, las melodías son increíbles"
  }'
```

**Response:**

```json
{
  "toxicity": {
    "label": "NON_TOXIC",
    "score_toxic": 0.0234,
    "score_non_toxic": 0.9766,
    "confidence": 0.9766
  },
  "spam": {
    "label": "NOT_SPAM",
    "score_spam": 0.0512,
    "score_not_spam": 0.9488,
    "confidence": 0.9488
  },
  "recommendation": "KEEP",
  "should_be_removed": false
}
```

#### Toxic Content Example

**Request:**

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Odio este álbum, el artista es un idiota"
  }'
```

**Response:**

```json
{
  "toxicity": {
    "label": "TOXIC",
    "score_toxic": 0.8923,
    "score_non_toxic": 0.1077,
    "confidence": 0.8923
  },
  "spam": {
    "label": "NOT_SPAM",
    "score_spam": 0.1234,
    "score_not_spam": 0.8766,
    "confidence": 0.8766
  },
  "recommendation": "REMOVE_TOXIC",
  "should_be_removed": true
}
```

#### Spam Content Example

**Request:**

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "¡Compra ahora! Descuentos increíbles en nuestra tienda. Haz click aquí: www.ejemplo.com"
  }'
```

**Response:**

```json
{
  "toxicity": {
    "label": "NON_TOXIC",
    "score_toxic": 0.1245,
    "score_non_toxic": 0.8755,
    "confidence": 0.8755
  },
  "spam": {
    "label": "SPAM",
    "score_spam": 0.9123,
    "score_not_spam": 0.0877,
    "confidence": 0.9123
  },
  "recommendation": "REMOVE_SPAM",
  "should_be_removed": true
}
```

#### Batch Classification

**Request:**

```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Excelente álbum, muy recomendado"},
    {"text": "Terrible música, el peor artista del mundo"},
    {"text": "Visita nuestra web para más ofertas"}
  ]'
```

**Response:**

```json
{
  "results": [
    {
      "text": "Excelente álbum, muy recomendado",
      "classification": {
        "toxicity": {
          "label": "NON_TOXIC",
          "score_toxic": 0.0156,
          "score_non_toxic": 0.9844,
          "confidence": 0.9844
        },
        "spam": {
          "label": "NOT_SPAM",
          "score_spam": 0.0423,
          "score_not_spam": 0.9577,
          "confidence": 0.9577
        },
        "recommendation": "KEEP",
        "should_be_removed": false
      }
    },
    {
      "text": "Terrible música, el peor artista del mundo",
      "classification": {
        "toxicity": {
          "label": "TOXIC",
          "score_toxic": 0.7856,
          "score_non_toxic": 0.2144,
          "confidence": 0.7856
        },
        "spam": {
          "label": "NOT_SPAM",
          "score_spam": 0.1023,
          "score_not_spam": 0.8977,
          "confidence": 0.8977
        },
        "recommendation": "REMOVE_TOXIC",
        "should_be_removed": true
      }
    },
    {
      "text": "Visita nuestra web para más ofertas",
      "classification": {
        "toxicity": {
          "label": "NON_TOXIC",
          "score_toxic": 0.0234,
          "score_non_toxic": 0.9766,
          "confidence": 0.9766
        },
        "spam": {
          "label": "SPAM",
          "score_spam": 0.8445,
          "score_not_spam": 0.1555,
          "confidence": 0.8445
        },
        "recommendation": "REMOVE_SPAM",
        "should_be_removed": true
      }
    }
  ]
}
```

### Health Check

**Request:**

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "hate_model_loaded": true,
  "spam_model_loaded": true,
  "device": "cuda:0"
}
```

## Response Schema

### ClassificationResponse

| Field                      | Type    | Description                                             |
| -------------------------- | ------- | ------------------------------------------------------- |
| `toxicity`                 | Object  | Toxicity classification results                         |
| `toxicity.label`           | String  | "TOXIC" or "NON_TOXIC"                                  |
| `toxicity.score_toxic`     | Float   | Probability of toxic content (0-1)                      |
| `toxicity.score_non_toxic` | Float   | Probability of non-toxic content (0-1)                  |
| `toxicity.confidence`      | Float   | Highest probability score                               |
| `spam`                     | Object  | Spam classification results                             |
| `spam.label`               | String  | "SPAM" or "NOT_SPAM"                                    |
| `spam.score_spam`          | Float   | Probability of spam content (0-1)                       |
| `spam.score_not_spam`      | Float   | Probability of legitimate content (0-1)                 |
| `spam.confidence`          | Float   | Highest probability score                               |
| `recommendation`           | String  | "KEEP", "REMOVE_TOXIC", "REMOVE_SPAM", or "REMOVE_BOTH" |
| `should_be_removed`        | Boolean | Whether the content should be removed                   |

## Configuration

### Environment Variables

The service can be configured using environment variables or a `.env` file:

```bash
# Application Settings
APP_NAME="Content Moderation ML Service"
APP_VERSION="1.0.0"
ENVIRONMENT="production"
DEBUG=false

# Server Settings
HOST="0.0.0.0"
PORT=8000

# Model Thresholds (0.0-1.0)
TOXIC_THRESHOLD=0.7  # Minimum confidence to flag as toxic
SPAM_THRESHOLD=0.7   # Minimum confidence to flag as spam

# Processing Limits
MAX_BATCH_SIZE=50    # Maximum reviews per batch request
MAX_TEXT_LENGTH=512  # Maximum text length in tokens

# ML Models (Hugging Face model IDs)
TOXICITY_MODEL="bgonzalezbustamante/bert-spanish-toxicity"
SPAM_MODEL="asfilcnx3/spam-detection-es"
```

### Configuration File

All settings are centralized in [app/core/config.py](app/core/config.py) using Pydantic Settings:

```python
from app.core.config import settings

# Access configuration
print(settings.toxic_threshold)  # 0.7
print(settings.max_batch_size)   # 50
```

### Adjusting Thresholds

To make the classifier more or less strict:

- **Higher threshold (0.8-0.9)**: Fewer false positives, may miss some toxic/spam content
- **Lower threshold (0.5-0.6)**: Catches more toxic/spam content, may have more false positives
- **Default (0.7)**: Balanced approach

## Deployment on Hugging Face Spaces

This service is designed to be deployed on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Select "Docker" as the SDK
3. Upload the project files
4. The service will automatically build and deploy

**Environment Variables:**

- No special environment variables required
- Models are automatically downloaded from Hugging Face on first run

**Hardware Recommendations:**

- CPU Basic: Works but slower inference (~1-2s per request)
- CPU Upgraded: Better performance (~0.5-1s per request)
- GPU: Recommended for production (~0.1-0.3s per request)

## Performance Considerations

- **First Request**: May take longer as models load into memory
- **Concurrent Requests**: Service supports multiple workers for parallel processing
- **Memory Usage**: Approximately 1.5-2GB RAM with both models loaded
- **GPU Acceleration**: Automatically detected and used when available

## Dependencies

- **FastAPI**: Web framework for building the API
- **Uvicorn**: ASGI server for running FastAPI
- **PyTorch**: Deep learning framework for model inference
- **Transformers**: Hugging Face library for transformer models
- **Pydantic**: Data validation using Python type annotations
- **Pydantic Settings**: Configuration management

## Development

### Code Organization

This project follows FastAPI best practices:

1. **Separation of Concerns**: API routes, business logic, and data models are in separate modules
2. **Dependency Injection**: Models are loaded once and reused (Singleton pattern)
3. **Type Hints**: Comprehensive type annotations throughout the codebase
4. **Documentation**: Detailed docstrings following Google style
5. **Pydantic Validation**: Automatic request/response validation

### API Documentation Standards

All endpoints include:

- **Summary**: Brief description of the endpoint
- **Description**: Detailed explanation with examples
- **Response Models**: Pydantic schemas with examples
- **Status Codes**: All possible HTTP status codes
- **Error Responses**: Example error responses

Access the interactive documentation at `/docs` (Swagger UI) or `/redoc` (ReDoc).

### Adding New Endpoints

To add a new endpoint:

1. Create a Pydantic schema in `app/schemas/`
2. Add business logic to `app/services/`
3. Create the route in `app/api/routes/`
4. Register the router in `app/main.py`

Example:

```python
# app/api/routes/example.py
from fastapi import APIRouter

router = APIRouter(prefix="/example", tags=["example"])

@router.get("/")
async def example_endpoint():
    return {"message": "Example"}

# app/main.py
from app.api.routes import example
app.include_router(example.router)
```

### Running Tests

#### Automated Testing

Use the provided test script to verify all endpoints:

```bash
# Start the service first
docker-compose up -d
# or
uvicorn app.main:app --reload

# Run the test script
./test_api.sh

# Test against a different URL
API_URL=http://your-server:8000 ./test_api.sh
```

The test script verifies:

- All health check endpoints
- Single and batch classification
- Input validation
- API documentation endpoints

#### Manual Testing

```bash
# Start the service
uvicorn app.main:app --reload

# In another terminal, test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Test review"}'
```

## Best Practices

### Production Deployment

1. **Environment Variables**: Use environment variables for sensitive configuration
2. **CORS**: Configure `allow_origins` in production (currently set to `*` for development)
3. **Logging**: Configure appropriate log levels
4. **Health Checks**: Use `/health`, `/readiness`, and `/liveness` endpoints
5. **Resource Limits**: Set appropriate CPU and memory limits in docker-compose.yml

### Security Considerations

- Service runs as non-root user in Docker
- Input validation through Pydantic models
- Text length limits prevent resource exhaustion
- Batch size limits prevent abuse
- CORS configured (adjust for production)

---

**Note**: This service processes text in Spanish. For optimal results, ensure input text is in Spanish.
