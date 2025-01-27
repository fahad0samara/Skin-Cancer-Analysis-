from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime
import uuid
from ..application.services import SkinCancerService, ModelTrainingService
from ..domain.interfaces import ImageData, PredictionResult, ModelMetrics
from ..infrastructure.repositories import (
    FileSystemImageRepository,
    FileSystemModelRepository,
    SQLitePredictionRepository
)

app = FastAPI(title="Skin Cancer Analysis API")

# Initialize repositories
image_repo = FileSystemImageRepository("data/images")
model_repo = FileSystemModelRepository("data/models")
prediction_repo = SQLitePredictionRepository("data/predictions.db")

# Initialize services
skin_cancer_service = SkinCancerService(image_repo, model_repo, prediction_repo)
training_service = ModelTrainingService(image_repo, model_repo)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a new image for analysis"""
    try:
        contents = await file.read()
        image_id = str(uuid.uuid4())
        
        # Create ImageData object
        image_data = ImageData(
            id=image_id,
            data=contents,
            label="unknown",
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "upload_time": datetime.now().isoformat()
            }
        )
        
        # Save image
        image_repo.save_image(image_data)
        
        return {"image_id": image_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{image_id}")
async def predict_image(
    image_id: str,
    model_id: str
) -> PredictionResult:
    """Make a prediction for a specific image"""
    try:
        return skin_cancer_service.predict_image(image_id, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(parameters: Dict[str, Any]) -> ModelMetrics:
    """Train a new model with the specified parameters"""
    try:
        return training_service.train_model(**parameters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
async def analyze_dataset() -> Dict[str, Any]:
    """Analyze the current dataset"""
    try:
        return skin_cancer_service.analyze_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str) -> ModelMetrics:
    """Get metrics for a specific model"""
    try:
        metrics = skin_cancer_service.get_model_metrics(model_id)
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
async def get_predictions(
    image_id: Optional[str] = None
) -> List[PredictionResult]:
    """Get prediction history for a specific image or all images"""
    try:
        return skin_cancer_service.get_prediction_history(image_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    """Start the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)
