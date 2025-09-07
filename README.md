# KrishokBondhu API - Docker Setup

This directory contains the FastAPI backend for KrishokBondhu with Docker support.

## Local Development with Docker

### Build the Docker image:

```bash
docker build -t krishokbondhu-api .
```

### Run with Docker:

```bash
docker run -p 8000:8000 --env-file .env krishokbondhu-api
```

### Run with Docker Compose:

```bash
docker-compose up --build
```

## Environment Variables

Create a `.env` file with the following variables:

```
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.1-70b-versatile
TEMPERATURE=0
MAX_TOKENS=1000
TIMEOUT=30
MAX_RETRIES=2
```

## Deployment

### Railway/Render:

1. Connect your GitHub repository
2. Set the root directory to `plantdisease_detection_api`
3. Add environment variables in the platform settings

### Google Cloud Run:

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/krishokbondhu-api
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/krishokbondhu-api --platform managed
```

### AWS ECR + ECS:

```bash
aws ecr create-repository --repository-name krishokbondhu-api
docker tag krishokbondhu-api:latest YOUR_ECR_URI:latest
docker push YOUR_ECR_URI:latest
```

## API Endpoints

- `GET /health` - Health check
- `GET /ping` - Simple ping
- `POST /disease-detection` - Plant disease detection
- `POST /recommendfertilizer` - Fertilizer recommendation
- `POST /recommendcrop` - Crop recommendation
- `POST /query` - LLM query for disease advice
- `POST /message` - Chat with agricultural assistant

## Model Files

Ensure these model files are present in the `model/` directory:

- `1.keras` - Disease detection model
- `RandomForest.pkl` - Crop recommendation model
- `fertilizerRecommendation.keras` - Fertilizer recommendation model
"# turfmania_backend" 
