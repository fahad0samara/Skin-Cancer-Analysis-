import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, DBSCAN
import shap
import lime
import lime.lime_tabular
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MulticlassSkinCancerAnalysis:
    def __init__(self, data_path, img_size=(224, 224)):
        self.data_path = data_path
        self.img_size = img_size
        self.features = None
        self.labels = None
        self.class_names = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """Load and preprocess image data"""
        features = []
        labels = []
        self.class_names = sorted(os.listdir(self.data_path))
        
        print("Loading images from classes:", self.class_names)
        
        for class_name in tqdm(self.class_names):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            features.append(img)
                            labels.append(class_name)
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
        
        self.features = np.array(features)
        self.labels = self.label_encoder.fit_transform(labels)
        
        print(f"Loaded {len(self.features)} images")
        print("Class distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(self.labels == i)
            print(f"{class_name}: {count} images")
        
        return self.features, self.labels
    
    def create_model(self):
        """Create EfficientNet model for transfer learning"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.class_names), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=10, batch_size=32):
        """Train the model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create model
        self.model = self.create_model()
        
        # Train model
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            validation_steps=len(X_test) // batch_size
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            test_datagen.flow(X_test, y_test, batch_size=batch_size),
            steps=len(X_test) // batch_size
        )
        
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Generate predictions
        y_pred = self.model.predict(test_datagen.flow(X_test, batch_size=batch_size))
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred_classes,
            target_names=self.class_names
        ))
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return history
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        # Class distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(x=self.labels)
        plt.title('Distribution of Classes')
        plt.xlabel('Class')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Image examples
        plt.figure(figsize=(15, 10))
        for i, class_idx in enumerate(np.unique(self.labels)):
            if i >= 9:  # Show max 9 examples
                break
            idx = np.where(self.labels == class_idx)[0][0]
            plt.subplot(3, 3, i+1)
            plt.imshow(self.features[idx])
            plt.title(self.class_names[class_idx])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Image size distribution
        sizes = [img.shape for img in self.features]
        plt.figure(figsize=(10, 5))
        plt.hist(sizes, bins=20)
        plt.title('Image Size Distribution')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def dimensionality_reduction(self):
        """Perform dimensionality reduction"""
        # Extract features using the trained model
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output
        )
        
        # Get features
        features = feature_extractor.predict(self.features / 255.0)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features)
        
        # UMAP
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(features)
        
        # Plot results
        plt.figure(figsize=(20, 5))
        
        # PCA plot
        plt.subplot(131)
        scatter = plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=self.labels,
            cmap='tab10'
        )
        plt.title('PCA')
        plt.colorbar(scatter)
        
        # t-SNE plot
        plt.subplot(132)
        scatter = plt.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=self.labels,
            cmap='tab10'
        )
        plt.title('t-SNE')
        plt.colorbar(scatter)
        
        # UMAP plot
        plt.subplot(133)
        scatter = plt.scatter(
            umap_result[:, 0],
            umap_result[:, 1],
            c=self.labels,
            cmap='tab10'
        )
        plt.title('UMAP')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'pca': pca_result,
            'tsne': tsne_result,
            'umap': umap_result
        }
    
    def analyze_predictions(self, num_samples=5):
        """Analyze model predictions on random samples"""
        # Get random samples
        indices = np.random.choice(len(self.features), num_samples, replace=False)
        samples = self.features[indices] / 255.0
        true_labels = self.labels[indices]
        
        # Get predictions
        predictions = self.model.predict(samples)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Plot results
        plt.figure(figsize=(15, 3*num_samples))
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(self.features[indices[i]])
            plt.title(f'True: {self.class_names[true_labels[i]]}\n' +
                     f'Predicted: {self.class_names[pred_labels[i]]} ' +
                     f'({predictions[i][pred_labels[i]]:.2%} confidence)')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def run_complete_analysis(self, epochs=10, batch_size=32):
        """Run complete analysis pipeline"""
        print("Loading and preprocessing data...")
        self.load_and_preprocess_data()
        
        print("\nPerforming EDA...")
        self.perform_eda()
        
        print("\nTraining model...")
        self.train_model(epochs=epochs, batch_size=batch_size)
        
        print("\nPerforming dimensionality reduction...")
        self.dimensionality_reduction()
        
        print("\nAnalyzing predictions...")
        self.analyze_predictions()
        
        return self.model

# Example usage:
if __name__ == "__main__":
    data_path = "path/to/train/data"
    analyzer = MulticlassSkinCancerAnalysis(data_path)
    model = analyzer.run_complete_analysis()
    analyzer.save_model("skin_cancer_model.h5")
