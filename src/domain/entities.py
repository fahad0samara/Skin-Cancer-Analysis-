from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

@dataclass
class Image:
    id: str
    path: str
    data: np.ndarray
    metadata: Dict[str, Any]
    diagnosis: Optional[str] = None
    confidence: Optional[float] = None
    
@dataclass
class AnalysisResult:
    image_id: str
    features: Dict[str, Any]
    predictions: Dict[str, float]
    analysis_date: datetime
    
@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    
@dataclass
class ClusteringResult:
    algorithm: str
    labels: np.ndarray
    silhouette_score: float
    parameters: Dict[str, Any]
    
@dataclass
class DimensionalityReductionResult:
    algorithm: str
    embedding: np.ndarray
    explained_variance_ratio: Optional[List[float]] = None
    
@dataclass
class FeatureImportance:
    feature_name: str
    importance_score: float
    model_name: str
    
@dataclass
class ModelExplanation:
    model_name: str
    feature_names: List[str]
    shap_values: np.ndarray
    lime_explanation: Dict[str, Any]
