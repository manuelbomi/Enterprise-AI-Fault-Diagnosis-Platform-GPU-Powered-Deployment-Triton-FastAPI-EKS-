from pydantic import BaseModel
from typing import Any, Dict, List

class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: Dict[str, Any]
    embeddings: List[float]
    similar_images: List[Dict[str, Any]]
