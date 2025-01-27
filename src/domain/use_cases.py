from typing import Dict, Any, List, Optional
from datetime import datetime
from .interfaces import (
    ImageRepository, ModelRepository, PredictionRepository,
    ImageData, PredictionResult, ModelMetrics
)

class TrainModelUseCase:
    def __init__(
        self,
        image_repository: ImageRepository,
        model_repository: ModelRepository
    ):
        self.image_repository = image_repository
        self.model_repository = model_repository
    
    def execute(self, parameters: Dict[str, Any]) -> ModelMetrics:
        # Get training data
        images = self.image_repository.get_all_images()
        
        # Train model (implementation in infrastructure layer)
        model, metrics = self._train_model(images, parameters)
        
        # Save model and metrics
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_repository.save_model(model_id, model)
        self.model_repository.save_metrics(metrics)
        
        return metrics
    
    def _train_model(
        self,
        images: List[ImageData],
        parameters: Dict[str, Any]
    ) -> tuple:
        # Implementation provided by infrastructure layer
        pass

class PredictImageUseCase:
    def __init__(
        self,
        image_repository: ImageRepository,
        model_repository: ModelRepository,
        prediction_repository: PredictionRepository
    ):
        self.image_repository = image_repository
        self.model_repository = model_repository
        self.prediction_repository = prediction_repository
    
    def execute(self, image_id: str, model_id: str) -> PredictionResult:
        # Get image and model
        image = self.image_repository.get_image(image_id)
        if not image:
            raise ValueError(f"Image {image_id} not found")
        
        model = self.model_repository.load_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Make prediction
        prediction = self._predict(image, model)
        
        # Save prediction
        self.prediction_repository.save_prediction(prediction)
        
        return prediction
    
    def _predict(self, image: ImageData, model: Any) -> PredictionResult:
        # Implementation provided by infrastructure layer
        pass

class AnalyzeDatasetUseCase:
    def __init__(self, image_repository: ImageRepository):
        self.image_repository = image_repository
    
    def execute(self) -> Dict[str, Any]:
        # Get all images
        images = self.image_repository.get_all_images()
        
        # Perform analysis
        analysis_results = self._analyze_dataset(images)
        
        return analysis_results
    
    def _analyze_dataset(self, images: List[ImageData]) -> Dict[str, Any]:
        # Implementation provided by infrastructure layer
        pass

class EvaluateModelUseCase:
    def __init__(
        self,
        image_repository: ImageRepository,
        model_repository: ModelRepository
    ):
        self.image_repository = image_repository
        self.model_repository = model_repository
    
    def execute(self, model_id: str) -> ModelMetrics:
        # Get model and test data
        model = self.model_repository.load_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        test_images = self.image_repository.get_all_images()
        
        # Evaluate model
        metrics = self._evaluate_model(model, test_images)
        
        # Save metrics
        self.model_repository.save_metrics(metrics)
        
        return metrics
    
    def _evaluate_model(
        self,
        model: Any,
        images: List[ImageData]
    ) -> ModelMetrics:
        # Implementation provided by infrastructure layer
        pass

class GetModelMetricsUseCase:
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
    
    def execute(self, model_id: str) -> Optional[ModelMetrics]:
        return self.model_repository.get_metrics(model_id)

class GetPredictionHistoryUseCase:
    def __init__(self, prediction_repository: PredictionRepository):
        self.prediction_repository = prediction_repository
    
    def execute(self, image_id: Optional[str] = None) -> List[PredictionResult]:
        if image_id:
            return self.prediction_repository.get_predictions_for_image(image_id)
        return self.prediction_repository.get_all_predictions()
