from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image
import os
from triton_client import TritonClient
from faiss_search import FaissSearcher
from models import PredictionResponse

app = FastAPI(title="Enterprise Fault FastAPI (Triton-backed)")

TRITON_URL = os.environ.get("TRITON_URL", "triton:8001")
TRITON_MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "vgg16")

triton = TritonClient(TRITON_URL)
faiss_searcher = FaissSearcher(index_path="/models/faiss/index.faiss", names_path="/models/faiss/image_names.npy")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224,224))
    arr = np.array(img).astype('float32')
    arr = arr[None, ...]
    return arr

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        tensor = preprocess_image(content)
        embeddings, probs = triton.infer_model(TRITON_MODEL_NAME, tensor)
        similar = faiss_searcher.search(embeddings, top_k=5)
        resp = PredictionResponse(
            predicted_label=probs.get('label','unknown'),
            probabilities=probs.get('probs',{}),
            embeddings=embeddings.flatten().tolist(),
            similar_images=[{"name": n, "score": s} for n,s in similar]
        )
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
