from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ImageData:
    id: str
    data: np.ndarray
    label: str
    metadata: Dict[str, Any]

@dataclass
class PredictionResult:
    image_id: str
    predicted_class: str
    confidence: float
    prediction_time: datetime

@dataclass
class ModelMetrics:
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    training_time: datetime

class ImageRepository(ABC):
    @abstractmethod
    def save_image(self, image: ImageData) -> None:
        pass
    
    @abstractmethod
    def get_image(self, image_id: str) -> Optional[ImageData]:
        pass
    
    @abstractmethod
    def get_all_images(self) -> List[ImageData]:
        pass
    
    @abstractmethod
    def get_images_by_class(self, class_name: str) -> List[ImageData]:
        pass

class ModelRepository(ABC):
    @abstractmethod
    def save_model(self, model_id: str, model_data: bytes) -> None:
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> Any:
        pass
    
    @abstractmethod
    def save_metrics(self, metrics: ModelMetrics) -> None:
        pass
    
    @abstractmethod
    def get_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        pass

class PredictionRepository(ABC):
    @abstractmethod
    def save_prediction(self, prediction: PredictionResult) -> None:
        pass
    
    @abstractmethod
    def get_predictions_for_image(self, image_id: str) -> List[PredictionResult]:
        pass
    
    @abstractmethod
    def get_all_predictions(self) -> List[PredictionResult]:
        pass

class AnalysisService(ABC):
    @abstractmethod
    def analyze_dataset(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def train_model(self, parameters: Dict[str, Any]) -> ModelMetrics:
        pass
    
    @abstractmethod
    def evaluate_model(self, model_id: str) -> ModelMetrics:
        pass
    
    @abstractmethod
    def predict_image(self, image_id: str) -> PredictionResult:
        pass
