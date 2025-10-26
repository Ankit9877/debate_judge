# api_server.py - FINAL CORRECTED VERSION
import os
import json
import requests
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# --- Config (FINAL CORRECTED PATHS) ---
HF_REPO_NAME = "Ankit18062005/debate-judge" 

# CORRECT: Use only the base repository ID. The subfolder is specified below.
MODEL_NAME_OR_PATH = HF_REPO_NAME 

# The scaling file path is correct.
SCALE_INFO_PATH = f"https://huggingface.co/{HF_REPO_NAME}/resolve/main/label_scale_info.json"
MAX_LENGTH = 256

# --- Pydantic Schemas (No Change) ---
class PredictionRequest(BaseModel):
    argument_text: str

class PredictionResponse(BaseModel):
    overall_score: float
    logical_score: float
    rhetorical_score: float
    all_scores: Dict[str, float] 

# --- Global Model and Data Variables (No Change) ---
MODEL = None
TOKENIZER = None
SCALE_INFO = None
LABEL_COLUMNS = None

# --- Core Functions (CRITICAL FINAL FIX FOR LOADING) ---
def load_model():
    """Load model, tokenizer, and scaling data once."""
    global MODEL, TOKENIZER, SCALE_INFO, LABEL_COLUMNS
    
    if MODEL is not None:
        return

    print(f"Loading model from Hugging Face Hub: {MODEL_NAME_OR_PATH}")
    # 1. Load Scaling Info (Correct and robust)
    try:
        response = requests.get(SCALE_INFO_PATH)
        response.raise_for_status() 
        SCALE_INFO = response.json()
        LABEL_COLUMNS = SCALE_INFO["label_columns"]
    except Exception as e:
        raise Exception(f"Failed to download scaling info from {SCALE_INFO_PATH}. Error: {e}")

    # 2. Load Tokenizer and Model (Uses subfolder argument to resolve path)
    # This resolves the HFValidationError
    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        subfolder="best_model"
    )
    MODEL = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH,
        subfolder="best_model"
    )
    
    # 3. Set device
    MODEL.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL.to(MODEL.device)
    MODEL.eval()
    print(f"Model loaded successfully on {MODEL.device} with {len(LABEL_COLUMNS)} criteria.")
    
def infer_texts(texts: List[str]) -> List[Dict[str, float]]:
    # ... (rest of the infer_texts function is unchanged) ...
    if MODEL is None:
        load_model()
    enc = TOKENIZER(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        logits = MODEL(**enc).logits.cpu().numpy()
    inv_preds = np.zeros_like(logits)
    for j, col in enumerate(LABEL_COLUMNS):
        min_c = SCALE_INFO["mins"][col]
        max_c = SCALE_INFO["maxs"][col]
        inv_preds[:, j] = logits[:, j] * (max_c - min_c) + min_c
    outputs = []
    for row in inv_preds:
        outputs.append({col: float(row[idx]) for idx, col in enumerate(LABEL_COLUMNS)})
    return outputs

# --- FastAPI App (No Change) ---
app = FastAPI()
API_KEY = os.environ.get("AI_API_KEY", "default_secret_key_change_me_in_prod")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict_scores(request_data: PredictionRequest, request: Request):
    # 1. Security Check 
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
    output_map = {
        "overall_score": score_data.get("Combined Quality", score_data.get("Combined_Quality", 0.0)),
        "logical_score": score_data.get("Logical Quality", score_data.get("Logical_Quality", 0.0)),
        "rhetorical_score": score_data.get("Rhetorical Quality", score_data.get("Rhetorical_Quality", 0.0)),
    }
    for k, v in output_map.items():
        output_map[k] = max(1.0, float(v))

    return PredictionResponse(
        overall_score=output_map["overall_score"],
        logical_score=output_map["logical_score"],
        rhetorical_score=output_map["rhetorical_score"],
        all_scores=score_data
    )
