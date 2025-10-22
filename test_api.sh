#!/bin/bash

# API Testing Script
# This script tests all endpoints of the Content Moderation ML Microservice

set -e

API_URL="${API_URL:-http://localhost:8000}"
echo "Testing API at: $API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
    fi
}

echo -e "${YELLOW}=== Testing Health Endpoints ===${NC}"

# Test 1: Root endpoint
echo "Test 1: GET /"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/")
if [ "$response" -eq 200 ]; then
    print_result 0 "Root endpoint"
else
    print_result 1 "Root endpoint (Got HTTP $response)"
fi
echo ""

# Test 2: Health endpoint
echo "Test 2: GET /health"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [ "$response" -eq 200 ]; then
    print_result 0 "Health endpoint"
    curl -s "$API_URL/health" | jq .
else
    print_result 1 "Health endpoint (Got HTTP $response)"
fi
echo ""

# Test 3: Readiness endpoint
echo "Test 3: GET /readiness"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/readiness")
if [ "$response" -eq 200 ]; then
    print_result 0 "Readiness endpoint"
else
    print_result 1 "Readiness endpoint (Got HTTP $response)"
fi
echo ""

# Test 4: Liveness endpoint
echo "Test 4: GET /liveness"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/liveness")
if [ "$response" -eq 200 ]; then
    print_result 0 "Liveness endpoint"
else
    print_result 1 "Liveness endpoint (Got HTTP $response)"
fi
echo ""

echo -e "${YELLOW}=== Testing Classification Endpoints ===${NC}"

# Test 5: Classify normal review
echo "Test 5: POST /classify (Normal review)"
response=$(curl -s -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Este álbum es una obra maestra, las melodías son increíbles"}')
if echo "$response" | jq -e '.recommendation' > /dev/null 2>&1; then
    print_result 0 "Normal review classification"
    echo "$response" | jq .
else
    print_result 1 "Normal review classification"
fi
echo ""

# Test 6: Classify toxic review
echo "Test 6: POST /classify (Toxic content)"
response=$(curl -s -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Odio este álbum, el artista es un idiota"}')
if echo "$response" | jq -e '.recommendation' > /dev/null 2>&1; then
    print_result 0 "Toxic content classification"
    echo "$response" | jq .
else
    print_result 1 "Toxic content classification"
fi
echo ""

# Test 7: Classify spam
echo "Test 7: POST /classify (Spam)"
response=$(curl -s -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "¡Compra ahora! Descuentos increíbles en nuestra tienda. Click aquí"}')
if echo "$response" | jq -e '.recommendation' > /dev/null 2>&1; then
    print_result 0 "Spam classification"
    echo "$response" | jq .
else
    print_result 1 "Spam classification"
fi
echo ""

# Test 8: Batch classification
echo "Test 8: POST /classify/batch"
response=$(curl -s -X POST "$API_URL/classify/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Excelente álbum, muy recomendado"},
    {"text": "Terrible música, el peor artista"},
    {"text": "Visita nuestra web para más ofertas"}
  ]')
if echo "$response" | jq -e '.total' > /dev/null 2>&1; then
    print_result 0 "Batch classification"
    echo "$response" | jq '.total, .successful, .failed'
else
    print_result 1 "Batch classification"
fi
echo ""

# Test 9: Empty text (should fail)
echo "Test 9: POST /classify (Empty text - should fail)"
response=$(curl -s -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}')
if echo "$response" | jq -e '.detail' > /dev/null 2>&1; then
    print_result 0 "Empty text validation"
else
    print_result 1 "Empty text validation"
fi
echo ""

echo -e "${YELLOW}=== Testing OpenAPI Documentation ===${NC}"

# Test 10: OpenAPI docs
echo "Test 10: GET /docs"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs")
if [ "$response" -eq 200 ]; then
    print_result 0 "Swagger UI documentation"
else
    print_result 1 "Swagger UI documentation (Got HTTP $response)"
fi
echo ""

# Test 11: ReDoc
echo "Test 11: GET /redoc"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/redoc")
if [ "$response" -eq 200 ]; then
    print_result 0 "ReDoc documentation"
else
    print_result 1 "ReDoc documentation (Got HTTP $response)"
fi
echo ""

# Test 12: OpenAPI JSON
echo "Test 12: GET /openapi.json"
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/openapi.json")
if [ "$response" -eq 200 ]; then
    print_result 0 "OpenAPI JSON schema"
else
    print_result 1 "OpenAPI JSON schema (Got HTTP $response)"
fi
echo ""

echo -e "${GREEN}=== All tests completed ===${NC}"
