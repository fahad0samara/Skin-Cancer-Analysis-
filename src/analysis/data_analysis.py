import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ..models.skin_cancer_model import SkinCancerDataset, SkinCancerModel
import os
from PIL import Image
import cv2
from tqdm import tqdm

class SkinCancerAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.num_classes = len(self.classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_dataset(self):
        """Analyze dataset distribution and characteristics"""
        class_counts = []
        image_sizes = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)
                class_counts.append(len(images))
                
                # Analyze image sizes for first 100 images
                for img_name in images[:100]:
                    img_path = os.path.join(class_dir, img_name)
                    img = Image.open(img_path)
                    image_sizes.append(img.size)
        
        # Plot class distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(x=self.classes, y=class_counts)
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot image size distribution
        sizes = np.array(image_sizes)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.hist(sizes[:, 0], bins=30)
        plt.title('Width Distribution')
        plt.xlabel('Width (pixels)')
        
        plt.subplot(122)
        plt.hist(sizes[:, 1], bins=30)
        plt.title('Height Distribution')
        plt.xlabel('Height (pixels)')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'class_counts': dict(zip(self.classes, class_counts)),
            'image_sizes': image_sizes
        }
    
    def train_and_evaluate(self, batch_size=32, num_epochs=10):
        """Train and evaluate the model"""
        # Create datasets and dataloaders
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = SkinCancerDataset(self.data_dir, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4)
        
        # Create model and training components
        model = SkinCancerModel(num_classes=self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        trained_model = SkinCancerModel.train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=num_epochs, device=self.device
        )
        
        return trained_model
    
    def analyze_predictions(self, model, num_samples=5):
        """Analyze model predictions on random samples"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Get random samples from each class
        plt.figure(figsize=(15, 3*num_samples))
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                images = os.listdir(class_dir)
                if images:
                    # Get random image
                    img_name = np.random.choice(images)
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Load and transform image
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    model.eval()
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Plot image and prediction
                    plt.subplot(len(self.classes), 1, i+1)
                    plt.imshow(image)
                    plt.title(f'True: {class_name}\n' +
                             f'Predicted: {self.classes[predicted_class]} ' +
                             f'({confidence:.2%})')
                    plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, batch_size=32, num_epochs=10):
        """Run complete analysis pipeline"""
        print("Analyzing dataset...")
        dataset_stats = self.analyze_dataset()
        
        print("\nTraining model...")
        model = self.train_and_evaluate(batch_size, num_epochs)
        
        print("\nAnalyzing predictions...")
        self.analyze_predictions(model)
        
        return {
            'dataset_stats': dataset_stats,
            'model': model
        }
