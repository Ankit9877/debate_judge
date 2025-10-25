# api_server.py
import os
import json
import requests # Used to download the scaling JSON from Hugging Face
from typing import List, Dict

import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# --- Config (MUST BE CORRECTLY SET) ---
# 1. Set the Hugging Face Repository Name
#    REPLACE THE PLACEHOLDER BELOW WITH YOUR ACTUAL HF REPO NAME
HF_REPO_NAME = "Ankit18062005/debate-judge" 

# 2. Set the Model Load Path (loads model and tokenizer directly from HF Hub)
MODEL_NAME_OR_PATH = HF_REPO_NAME

# 3. Set the Scaling Info Path (loads JSON via HTTP request from the Hugging Face raw content URL)
#    Note: Assumes the file is saved in the 'distilbert_model_for_api/distilbert_webis_out/' path within your HF repo.
SCALE_INFO_PATH = f"https://huggingface.co/{HF_REPO_NAME}/raw/main/distilbert_model_for_api/distilbert_webis_out/label_scale_info.json"
MAX_LENGTH = 256

# --- Pydantic Schemas ---
class PredictionRequest(BaseModel):
    argument_text: str

class PredictionResponse(BaseModel):
    # These scores must be on the 1-5 scale for the Supabase function to work
    overall_score: float
    logical_score: float
    rhetorical_score: float
    all_scores: Dict[str, float] 

# --- Global Model and Data Variables ---
MODEL = None
TOKENIZER = None
SCALE_INFO = None
LABEL_COLUMNS = None

# --- Core Functions ---
def load_model():
    """Load model, tokenizer, and scaling data once."""
    global MODEL, TOKENIZER, SCALE_INFO, LABEL_COLUMNS
    
    if MODEL is not None:
        return

    print(f"Loading model from Hugging Face Hub: {MODEL_NAME_OR_PATH}")
    # 1. Load Scaling Info (Now downloaded from the Hub)
    try:
        response = requests.get(SCALE_INFO_PATH)
        response.raise_for_status() 
        SCALE_INFO = response.json()
        LABEL_COLUMNS = SCALE_INFO["label_columns"]
    except Exception as e:
        raise Exception(f"Failed to download scaling info from {SCALE_INFO_PATH}. Error: {e}")

    # 2. Load Tokenizer and Model (Loaded directly from the Hub)
    # The transformer library handles downloading the files from the HF Hub based on the repo name.
    TOKENIZER = DistilBertTokenizerFast.from_pretrained(MODEL_NAME_OR_PATH)
    MODEL = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
    
    # 3. Set device
    MODEL.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL.to(MODEL.device)
    MODEL.eval()
    print(f"Model loaded successfully on {MODEL.device} with {len(LABEL_COLUMNS)} criteria.")
    
def infer_texts(texts: List[str]) -> List[Dict[str, float]]:
    """Perform inference and inverse-scale results."""
    # Ensure model and tokenizer are loaded
    if MODEL is None:
        load_model()
        
    enc = TOKENIZER(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(MODEL.device)
    
    with torch.no_grad():
        logits = MODEL(**enc).logits.cpu().numpy()
        
    inv_preds = np.zeros_like(logits)
    
    # Inverse Scale: Un-normalize from [0, 1] back to original scale (e.g., 1-5)
    for j, col in enumerate(LABEL_COLUMNS):
        min_c = SCALE_INFO["mins"][col]
        max_c = SCALE_INFO["maxs"][col]
        inv_preds[:, j] = logits[:, j] * (max_c - min_c) + min_c
        
    outputs = []
    for row in inv_preds:
        outputs.append({col: float(row[idx]) for idx, col in enumerate(LABEL_COLUMNS)})
    return outputs

# --- FastAPI App ---
app = FastAPI()

# !!! IMPORTANT: This MUST match the AI_API_KEY set in your Render and Supabase secrets !!!
API_KEY = os.environ.get("AI_API_KEY", "default_secret_key_change_me_in_prod")

@app.on_event("startup")
async def startup_event():
    """Load the model when the FastAPI app starts."""
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict_scores(request_data: PredictionRequest, request: Request):
    """API endpoint to receive debate text and return scores."""
    # 1. Security Check (Authorization header is needed for the Supabase function)
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.split(" ")[-1] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    
    # 2. Inference
    text = [request_data.argument_text]
    results = infer_texts(text)
    
    if not results:
        raise HTTPException(status_code=500, detail="Inference failed.")
    
    # 3. Format Output for Supabase Function
    score_data = results[0]
    
    # Map criteria names to the expected output keys
    output_map = {
        "overall_score": score_data.get("Combined Quality", score_data.get("Combined_Quality", 0.0)),
        "logical_score": score_data.get("Logical Quality", score_data.get("Logical_Quality", 0.0)),
        "rhetorical_score": score_data.get("Rhetorical Quality", score_data.get("Rhetorical_Quality", 0.0)),
    }
    
    # Clip negative or very low scores to 1.0
    for k, v in output_map.items():
        output_map[k] = max(1.0, float(v))

    return PredictionResponse(
        overall_score=output_map["overall_score"],
        logical_score=output_map["logical_score"],
        rhetorical_score=output_map["rhetorical_score"],
        all_scores=score_data
    )