from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import pickle
from datetime import datetime, timedelta
import json

# app/services.py
from langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow all origins (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of origins, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the ChatGroq client
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=float(os.getenv("TEMPERATURE", 0)),
    max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
    timeout=int(os.getenv("TIMEOUT", 30)),
    max_retries=int(os.getenv("MAX_RETRIES", 2)),
)

def get_llm_response(query: str) -> str:
    response = llm.invoke(query)
    return response.content

# Global model variables
MODEL = None
crop_model = None

# Load models
def load_disease_model():
    global MODEL
    if MODEL is None:
        try:
            MODEL = tf.keras.models.load_model("./model/1.keras")
            logger.info("Disease detection model loaded successfully")
        except Exception as e:
            logger.warning(f"Disease model not found: {e}. Disease detection will be disabled.")
            MODEL = None

def load_crop_model():
    global crop_model
    if crop_model is None:
        try:
            with open("./model/RandomForest.pkl", "rb") as file:
                crop_model = pickle.load(file)
            logger.info("Crop recommendation model loaded successfully")
        except Exception as e:
            logger.warning(f"Crop model not found: {e}. Crop recommendation will be disabled.")
            crop_model = None

# Load models at startup
load_disease_model()
load_crop_model()

# Class names for disease detection
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Pydantic models for requests
class CropPredictionRequest(BaseModel):
    N: float
    P: float  
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"

class QueryRequest(BaseModel):
    query: str

# Helper function to read uploaded image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((224, 224))  # Resize to model's expected input size
    image = np.array(image)
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    return image / 255.0  # Normalize pixel values

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "KrishokBondhu API is running",
        "endpoints": {
            "health": "/health",
            "ping": "/ping", 
            "disease_detection": "/disease-detection",
            "crop_recommendation": "/recommendcrop",
            "query": "/query",
            "chat": "/message"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "disease_model": MODEL is not None,
            "crop_model": crop_model is not None
        }
    }

# Ping endpoint
@app.get("/ping")
async def ping():
    return {"message": "pong"}

# Prediction route
@app.post("/disease-detection")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Disease detection model is not available")
    
    # Read and preprocess image
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)  # Expand dimensions for batch processing

    # Make prediction
    prediction = MODEL.predict(image_batch)
    predicted_class_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # Get class label
    predicted_class = CLASS_NAMES[predicted_class_index]

    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }

@app.post("/recommendcrop")
async def predict_crop(data: CropPredictionRequest):
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop prediction model is not available")
    
    input_data = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    prediction = crop_model.predict(input_data)
    return {"prediction": prediction[0]}

# Chat storage (in-memory for demonstration)
chat_history = {}

@app.post("/message")
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message
    
    # Initialize chat history for new users
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    # Add user message to history
    chat_history[user_id].append({"role": "user", "content": user_message, "timestamp": datetime.now()})
    
    # Build context from recent chat history (last 10 messages)
    recent_history = chat_history[user_id][-10:]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    # Enhanced system prompt for agricultural context
    system_prompt = """You are KrishokBondhu, an expert agricultural assistant helping farmers in Bangladesh. 
    You provide practical advice on:
    - Crop cultivation and farming techniques
    - Plant disease identification and treatment
    - Soil management and fertilizer recommendations  
    - Weather-related farming decisions
    - Pest control methods
    - Crop rotation and seasonal planning
    - Market information and pricing
    
    Always provide actionable, region-specific advice considering Bangladesh's climate and farming practices.
    Keep responses concise but informative. If you're unsure, recommend consulting local agricultural experts.
    
    Chat History:
    {context}
    
    Current Question: {user_message}"""
    
    try:
        # Get LLM response
        prompt = system_prompt.format(context=context, user_message=user_message)
        bot_response = get_llm_response(prompt)
        
        # Add bot response to history
        chat_history[user_id].append({"role": "assistant", "content": bot_response, "timestamp": datetime.now()})
        
        return {
            "response": bot_response,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Sorry, I'm having trouble responding right now. Please try again.")

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # Enhanced system prompt for agricultural queries
        system_prompt = f"""You are KrishokBondhu, an expert agricultural assistant for farmers in Bangladesh. 
        Provide practical, actionable advice for this farming question: {request.query}
        
        Focus on:
        - Solutions specific to Bangladesh's climate and soil conditions
        - Cost-effective methods suitable for small to medium farmers
        - Seasonal considerations and timing
        - Local availability of resources and materials
        - Preventive measures and best practices
        
        Keep the response concise but comprehensive."""
        
        response = get_llm_response(system_prompt)
        
        return {
            "query": request.query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail="Sorry, I couldn't process your query right now. Please try again.")
