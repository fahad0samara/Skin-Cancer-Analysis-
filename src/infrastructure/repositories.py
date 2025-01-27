import os
import json
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import sqlite3
from ..domain.interfaces import (
    ImageRepository, ModelRepository, PredictionRepository,
    ImageData, PredictionResult, ModelMetrics
)

class FileSystemImageRepository(ImageRepository):
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.metadata_path = os.path.join(base_path, "metadata")
        os.makedirs(self.metadata_path, exist_ok=True)
    
    def save_image(self, image: ImageData) -> None:
        # Save image data as numpy array
        img_path = os.path.join(self.base_path, f"{image.id}.npy")
        np.save(img_path, image.data)
        
        # Save metadata
        metadata = {
            "id": image.id,
            "label": image.label,
            "metadata": image.metadata
        }
        
        with open(os.path.join(self.metadata_path, f"{image.id}.json"), "w") as f:
            json.dump(metadata, f)
    
    def get_image(self, image_id: str) -> Optional[ImageData]:
        try:
            # Load metadata
            with open(os.path.join(self.metadata_path, f"{image_id}.json"), "r") as f:
                metadata = json.load(f)
            
            # Load image data
            img_path = os.path.join(self.base_path, f"{image_id}.npy")
            img_data = np.load(img_path)
            
            return ImageData(
                id=metadata["id"],
                data=img_data,
                label=metadata["label"],
                metadata=metadata["metadata"]
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def get_all_images(self) -> List[ImageData]:
        images = []
        for filename in os.listdir(self.metadata_path):
            if filename.endswith(".json"):
                image_id = filename[:-5]
                image = self.get_image(image_id)
                if image:
                    images.append(image)
        return images
    
    def get_images_by_class(self, class_name: str) -> List[ImageData]:
        return [
            img for img in self.get_all_images()
            if img.label == class_name
        ]

class SQLitePredictionRepository(PredictionRepository):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    prediction_time TIMESTAMP NOT NULL
                )
            """)
    
    def save_prediction(self, prediction: PredictionResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions
                (image_id, predicted_class, confidence, prediction_time)
                VALUES (?, ?, ?, ?)
            """, (
                prediction.image_id,
                prediction.predicted_class,
                prediction.confidence,
                prediction.prediction_time.isoformat()
            ))
    
    def get_predictions_for_image(self, image_id: str) -> List[PredictionResult]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT image_id, predicted_class, confidence, prediction_time
                FROM predictions
                WHERE image_id = ?
                ORDER BY prediction_time DESC
            """, (image_id,))
            
            return [
                PredictionResult(
                    image_id=row[0],
                    predicted_class=row[1],
                    confidence=row[2],
                    prediction_time=datetime.fromisoformat(row[3])
                )
                for row in cursor.fetchall()
            ]
    
    def get_all_predictions(self) -> List[PredictionResult]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT image_id, predicted_class, confidence, prediction_time
                FROM predictions
                ORDER BY prediction_time DESC
            """)
            
            return [
                PredictionResult(
                    image_id=row[0],
                    predicted_class=row[1],
                    confidence=row[2],
                    prediction_time=datetime.fromisoformat(row[3])
                )
                for row in cursor.fetchall()
            ]

class FileSystemModelRepository(ModelRepository):
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.models_path = os.path.join(base_path, "models")
        self.metrics_path = os.path.join(base_path, "metrics")
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.metrics_path, exist_ok=True)
    
    def save_model(self, model_id: str, model_data: bytes) -> None:
        model_path = os.path.join(self.models_path, f"{model_id}.pkl")
        with open(model_path, "wb") as f:
            f.write(model_data)
    
    def load_model(self, model_id: str) -> Any:
        try:
            model_path = os.path.join(self.models_path, f"{model_id}.pkl")
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
    
    def save_metrics(self, metrics: ModelMetrics) -> None:
        metrics_dict = {
            "model_id": metrics.model_id,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "confusion_matrix": metrics.confusion_matrix.tolist(),
            "training_time": metrics.training_time.isoformat()
        }
        
        metrics_path = os.path.join(
            self.metrics_path,
            f"{metrics.model_id}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f)
    
    def get_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        try:
            metrics_path = os.path.join(self.metrics_path, f"{model_id}.json")
            with open(metrics_path, "r") as f:
                metrics_dict = json.load(f)
            
            return ModelMetrics(
                model_id=metrics_dict["model_id"],
                accuracy=metrics_dict["accuracy"],
                precision=metrics_dict["precision"],
                recall=metrics_dict["recall"],
                f1_score=metrics_dict["f1_score"],
                confusion_matrix=np.array(metrics_dict["confusion_matrix"]),
                training_time=datetime.fromisoformat(metrics_dict["training_time"])
            )
        except FileNotFoundError:
            return None
