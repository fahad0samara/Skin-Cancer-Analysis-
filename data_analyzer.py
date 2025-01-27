import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, accuracy_score
import torch
from torchvision import transforms
from collections import defaultdict

class DataAnalyzer:
    def __init__(self):
        self.train_dir = "Train"
        self.test_dir = "Test"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def get_dataset_stats(self):
        """Get basic statistics about the dataset"""
        stats = {
            'total_images': 0,
            'benign_count': 0,
            'malignant_count': 0,
            'image_sizes': []
        }
        
        # Process training directory
        for class_name in os.listdir(self.train_dir):
            class_dir = os.path.join(self.train_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        stats['total_images'] += 1
                        stats['image_sizes'].append(img.size[0] * img.size[1])
                        
                        if 'benign' in class_name.lower():
                            stats['benign_count'] += 1
                        else:
                            stats['malignant_count'] += 1
                    except:
                        continue
        
        return stats

    def get_sample_images(self, n=5):
        """Get n sample images from each class"""
        samples = []
        
        for class_name in os.listdir(self.train_dir):
            class_dir = os.path.join(self.train_dir, class_name)
            if os.path.isdir(class_dir):
                img_files = os.listdir(class_dir)[:n]
                for img_name in img_files:
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        img = img.resize((224, 224))
                        samples.append((img, class_name))
                    except:
                        continue
        
        return samples

    def get_model_metrics(self):
        """Calculate model performance metrics"""
        # This would typically use your trained model to make predictions
        # For now, we'll return sample metrics
        metrics = {
            'accuracy': 85.5,
            'precision': 83.2,
            'recall': 87.8,
            'confusion_matrix': np.array([[150, 20], [25, 175]]),
            'fpr': np.linspace(0, 1, 100),
            'tpr': np.linspace(0, 1, 100) ** 2,  # Sample ROC curve
            'auc': 0.89
        }
        
        return metrics

    def get_abcd_statistics(self):
        """Calculate statistics for ABCD criteria"""
        stats = {
            'asymmetry': np.random.normal(0.5, 0.2, 100),  # Sample data
            'border': np.random.normal(0.6, 0.15, 100),
            'color': np.random.normal(0.4, 0.25, 100),
            'diameter': np.random.normal(5, 2, 100)
        }
        
        return stats

    def get_abcd_correlation(self):
        """Calculate correlation between ABCD criteria"""
        # Sample correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.3, 0.4, 0.2],
            [0.3, 1.0, 0.5, 0.3],
            [0.4, 0.5, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0]
        ])
        
        return correlation_matrix

    def analyze_image(self, image):
        """Perform detailed analysis on a single image"""
        # Convert image to tensor
        if isinstance(image, str):
            image = Image.open(image)
        
        tensor_image = self.transform(image)
        
        # Calculate basic image statistics
        stats = {
            'mean': tensor_image.mean().item(),
            'std': tensor_image.std().item(),
            'min': tensor_image.min().item(),
            'max': tensor_image.max().item(),
            'size': image.size
        }
        
        return stats

    def get_training_progress(self):
        """Get training progress metrics"""
        # This would typically read from training logs
        # For now, return sample data
        epochs = 10
        progress = {
            'loss_history': np.random.exponential(0.5, epochs)[::-1],  # Decreasing loss
            'accuracy_history': 100 * (1 - np.exp(-np.linspace(0, 2, epochs))),  # Increasing accuracy
            'current_epoch': epochs,
            'total_epochs': epochs
        }
        
        return progress

    def get_class_activation_maps(self, image):
        """Generate class activation maps for visualization"""
        # This would typically use your model's attention mechanisms
        # For now, return a sample heatmap
        heatmap = np.random.rand(7, 7)  # Sample 7x7 attention map
        return heatmap

    def get_feature_importance(self):
        """Calculate feature importance scores"""
        features = ['Asymmetry', 'Border', 'Color', 'Diameter']
        importance = np.random.rand(4)
        importance = importance / importance.sum()
        
        return dict(zip(features, importance))
