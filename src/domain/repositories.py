from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .entities import (
    Image, AnalysisResult, ModelMetrics, ClusteringResult,
    DimensionalityReductionResult, FeatureImportance, ModelExplanation
)

class ImageRepository(ABC):
    @abstractmethod
    def save(self, image: Image) -> None:
        pass
    
    @abstractmethod
    def get_by_id(self, image_id: str) -> Optional[Image]:
        pass
    
    @abstractmethod
    def get_all(self) -> List[Image]:
        pass
    
    @abstractmethod
    def update(self, image: Image) -> None:
        pass
    
    @abstractmethod
    def delete(self, image_id: str) -> None:
        pass

class AnalysisRepository(ABC):
    @abstractmethod
    def save_result(self, result: AnalysisResult) -> None:
        pass
    
    @abstractmethod
    def get_result_by_image_id(self, image_id: str) -> Optional[AnalysisResult]:
        pass
    
    @abstractmethod
    def get_all_results(self) -> List[AnalysisResult]:
        pass
    
    @abstractmethod
    def update_result(self, result: AnalysisResult) -> None:
        pass

class ModelRepository(ABC):
    @abstractmethod
    def save_metrics(self, metrics: ModelMetrics) -> None:
        pass
    
    @abstractmethod
    def get_metrics_by_model(self, model_name: str) -> Optional[ModelMetrics]:
        pass
    
    @abstractmethod
    def get_all_metrics(self) -> List[ModelMetrics]:
        pass
    
    @abstractmethod
    def save_model(self, model_name: str, model_data: bytes) -> None:
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> Any:
        pass

class MLAnalysisRepository(ABC):
    @abstractmethod
    def save_clustering_result(self, result: ClusteringResult) -> None:
        pass
    
    @abstractmethod
    def save_dimensionality_reduction(self, result: DimensionalityReductionResult) -> None:
        pass
    
    @abstractmethod
    def save_feature_importance(self, importance: List[FeatureImportance]) -> None:
        pass
    
    @abstractmethod
    def save_model_explanation(self, explanation: ModelExplanation) -> None:
        pass
    
    @abstractmethod
    def get_clustering_results(self) -> List[ClusteringResult]:
        pass
    
    @abstractmethod
    def get_dimensionality_reductions(self) -> List[DimensionalityReductionResult]:
        pass
    
    @abstractmethod
    def get_feature_importance(self, model_name: str) -> List[FeatureImportance]:
        pass
    
    @abstractmethod
    def get_model_explanation(self, model_name: str) -> Optional[ModelExplanation]:
        pass
