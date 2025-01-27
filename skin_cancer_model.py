import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define which classes are considered malignant
MALIGNANT_CLASSES = {
    'melanoma',
    'basal cell carcinoma',
    'squamous cell carcinoma'
}

class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return a blank image and the label if there's an error
            blank_image = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224))
            return blank_image, label

class SkinCancerModel:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model = self.create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )

    def create_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
        return model

    def load_data(self, data_dir):
        image_paths = []
        labels = []
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                # Determine if this class is malignant
                is_malignant = class_name.lower() in MALIGNANT_CLASSES
                
                for image_name in os.listdir(class_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_dir, image_name)
                        image_paths.append(image_path)
                        labels.append(1 if is_malignant else 0)
        
        return image_paths, labels

    def train_model(self, train_loader, num_epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
        print(f"Training on device: {device}")
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            self.scheduler.step(epoch_loss)
        
        return history

    def evaluate_model(self, test_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                 target_names=['Benign', 'Malignant']))
        
        return predictions, true_labels

    def predict_image(self, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.test_transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Perform ABCD analysis
        mask = self.segment_lesion(image)
        abcd_results = {
            'asymmetry': self.analyze_asymmetry(mask),
            'border': self.analyze_border(mask),
            'color': self.analyze_color(image, mask),
            'diameter': self.analyze_diameter(mask)
        }
        
        return {
            'prediction': prediction.item(),
            'probabilities': probabilities.squeeze().cpu().numpy(),
            'abcd_results': abcd_results,
            'mask': mask
        }

    def analyze_asymmetry(self, mask):
        h, w = mask.shape
        left = mask[:, :w//2]
        right = np.fliplr(mask[:, w//2:])
        asymmetry = np.sum(np.abs(left - right)) / (h * w/2)
        return float(asymmetry)

    def analyze_border(self, mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return 1 - circularity
        return 0.0

    def analyze_color(self, image, mask):
        if isinstance(image, Image.Image):
            image = np.array(image)
        masked = cv2.bitwise_and(image, image, mask=mask)
        valid_pixels = masked[mask > 0]
        if len(valid_pixels) > 0:
            std_devs = np.std(valid_pixels, axis=0)
            return float(np.mean(std_devs) / 255.0)
        return 0.0

    def analyze_diameter(self, mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            _, radius = cv2.minEnclosingCircle(contours[0])
            diameter_mm = 2 * radius * 0.1  # assuming 1 pixel ≈ 0.1mm
            return float(diameter_mm)
        return 0.0

    def segment_lesion(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def plot_training_results(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'])
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()

    def visualize_prediction(self, image_path, results):
        image = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(results['mask'], cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.bar(['Benign', 'Malignant'], results['probabilities'])
        plt.title('Prediction Probabilities')
        
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """Save the model state to a file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model state from a file"""
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False
            
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")
        return True

    def get_prediction_explanation(self, abcd_results):
        """Generate a detailed explanation of the ABCD analysis"""
        explanation = []
        
        # Asymmetry analysis
        if abcd_results['asymmetry'] > 0.5:
            explanation.append("High asymmetry detected, which is a concerning factor.")
        else:
            explanation.append("The lesion appears fairly symmetric.")
            
        # Border analysis
        if abcd_results['border'] > 0.5:
            explanation.append("The border is irregular, which could be a sign of malignancy.")
        else:
            explanation.append("The border appears regular and well-defined.")
            
        # Color analysis
        if abcd_results['color'] > 0.3:
            explanation.append("Multiple color variations detected, which needs attention.")
        else:
            explanation.append("The color appears relatively uniform.")
            
        # Diameter analysis
        if abcd_results['diameter'] > 6:  # 6mm is often used as a threshold
            explanation.append(f"The diameter is approximately {abcd_results['diameter']:.1f}mm, "
                            "which is larger than 6mm and should be examined.")
        else:
            explanation.append(f"The diameter is approximately {abcd_results['diameter']:.1f}mm, "
                            "which is within normal range.")
            
        return explanation

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = SkinCancerModel()
    
    # Set data directories
    train_dir = "Train"
    test_dir = "Test"
    
    # Load training data
    train_paths, train_labels = model.load_data(train_dir)
    
    # Create dataset and dataloader
    train_dataset = SkinCancerDataset(train_paths, train_labels, transform=model.transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train model
    history = model.train_model(train_loader)
    
    # Plot training results
    model.plot_training_results(history)
    
    # Make prediction on a sample image
    sample_image_path = os.path.join(test_dir, "sample_image.jpg")  # Replace with actual path
    if os.path.exists(sample_image_path):
        results = model.predict_image(sample_image_path)
        model.visualize_prediction(sample_image_path, results)
        
        print(f"Prediction: {'Malignant' if results['prediction'] == 1 else 'Benign'}")
        print(f"Probabilities: Benign: {results['probabilities'][0]:.2f}, "
              f"Malignant: {results['probabilities'][1]:.2f}")
        print("\nABCD Analysis Results:")
        for key, value in results['abcd_results'].items():
            print(f"{key.capitalize()}: {value:.3f}")
