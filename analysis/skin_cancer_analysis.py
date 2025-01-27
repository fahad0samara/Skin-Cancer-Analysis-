import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
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

class SkinCancerAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.features = None
        self.labels = None
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess image data"""
        features = []
        labels = []
        
        # Load benign images
        benign_path = os.path.join(self.data_path, 'benign')
        for img_name in tqdm(os.listdir(benign_path)):
            img_path = os.path.join(benign_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                features.append(img.flatten())
                labels.append(0)
        
        # Load malignant images
        malignant_path = os.path.join(self.data_path, 'malignant')
        for img_name in tqdm(os.listdir(malignant_path)):
            img_path = os.path.join(malignant_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                features.append(img.flatten())
                labels.append(1)
        
        self.features = np.array(features)
        self.labels = np.array(labels)
        
        return self.features, self.labels
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        # Class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.labels)
        plt.title('Distribution of Classes')
        plt.xlabel('Class (0: Benign, 1: Malignant)')
        plt.show()
        
        # Feature distribution
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            sns.kdeplot(data=self.features[:, i], hue=self.labels)
            plt.title(f'Feature {i+1} Distribution')
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        corr_matrix = np.corrcoef(self.features.T)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix[:10, :10], annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Matrix (First 10 Features)')
        plt.show()
    
    def train_models(self):
        """Train and compare different models"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # ROC curve
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.show()
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        
        # Compare model performances
        results_df = pd.DataFrame(results).T
        plt.figure(figsize=(12, 6))
        results_df.plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return results
    
    def dimensionality_reduction(self):
        """Perform dimensionality reduction"""
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(15, 5))
        
        # PCA plot
        plt.subplot(131)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.labels, cmap='viridis')
        plt.title('PCA')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features_scaled)
        
        plt.subplot(132)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.labels, cmap='viridis')
        plt.title('t-SNE')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        
        # UMAP
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(features_scaled)
        
        plt.subplot(133)
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c=self.labels, cmap='viridis')
        plt.title('UMAP')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        
        plt.tight_layout()
        plt.show()
        
        return {'pca': pca_result, 'tsne': tsne_result, 'umap': umap_result}
    
    def clustering_analysis(self):
        """Perform clustering analysis"""
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # K-means
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features_scaled)
        
        plt.figure(figsize=(15, 5))
        
        # Plot K-means results
        plt.subplot(121)
        plt.scatter(features_scaled[:, 0], features_scaled[:, 1], 
                   c=kmeans_labels, cmap='viridis')
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # Plot DBSCAN results
        plt.subplot(122)
        plt.scatter(features_scaled[:, 0], features_scaled[:, 1], 
                   c=dbscan_labels, cmap='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
        
        return {'kmeans': kmeans_labels, 'dbscan': dbscan_labels}
    
    def feature_importance(self):
        """Analyze feature importance"""
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.features, self.labels)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Plot top 20 important features
        plt.figure(figsize=(12, 6))
        feature_importance = pd.DataFrame({
            'feature': range(len(importance)),
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance.head(20), x='feature', y='importance')
        plt.title('Top 20 Important Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def model_explanation(self):
        """Generate model explanations using SHAP and LIME"""
        # Train a simple model for explanations
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.features, self.labels)
        
        # SHAP explanations
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(self.features[:100])  # Use first 100 samples
        
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, self.features[:100], plot_type='bar')
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # LIME explanation for a single instance
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.features,
            mode='classification',
            training_labels=self.labels,
            feature_names=[f'feature_{i}' for i in range(self.features.shape[1])]
        )
        
        # Explain first instance
        exp = explainer.explain_instance(
            self.features[0], 
            rf.predict_proba,
            num_features=10
        )
        
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title('LIME Explanation for Single Instance')
        plt.tight_layout()
        plt.show()
        
        return {'shap_values': shap_values}

    def run_complete_analysis(self):
        """Run all analyses"""
        print("Loading and preprocessing data...")
        self.load_and_preprocess_data()
        
        print("\nPerforming EDA...")
        self.perform_eda()
        
        print("\nTraining and comparing models...")
        model_results = self.train_models()
        
        print("\nPerforming dimensionality reduction...")
        dim_reduction_results = self.dimensionality_reduction()
        
        print("\nPerforming clustering analysis...")
        clustering_results = self.clustering_analysis()
        
        print("\nAnalyzing feature importance...")
        importance_results = self.feature_importance()
        
        print("\nGenerating model explanations...")
        explanation_results = self.model_explanation()
        
        return {
            'model_results': model_results,
            'dim_reduction_results': dim_reduction_results,
            'clustering_results': clustering_results,
            'importance_results': importance_results,
            'explanation_results': explanation_results
        }
