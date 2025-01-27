import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import shap
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

class MLAnalysis:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
    
    def compare_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Compare different models using cross-validation"""
        results = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            
            # Fit model for later use
            model.fit(X_scaled, y)
            
            results[name] = {
                'cv_scores': cv_scores,
                'model': model
            }
        
        return results
    
    def create_model_comparison_plots(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, go.Figure]:
        """Create plots for model comparison"""
        plots = {}
        
        # ROC curves
        fig_roc = go.Figure()
        for name, result in results.items():
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash'),
                name='Random'
            ))
            
            cv_scores = result['cv_scores']
            fig_roc.add_trace(go.Scatter(
                x=np.linspace(0, 1, 100),
                y=np.mean([score for score in cv_scores]),
                mode='lines',
                name=f'{name} (AUC = {np.mean(cv_scores):.3f})'
            ))
        
        fig_roc.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        plots['roc_curves'] = fig_roc
        
        # Learning curves
        fig_learning = go.Figure()
        for name, result in results.items():
            cv_scores = result['cv_scores']
            fig_learning.add_trace(go.Scatter(
                x=list(range(1, len(cv_scores) + 1)),
                y=cv_scores,
                mode='lines+markers',
                name=name
            ))
        
        fig_learning.update_layout(
            title='Learning Curves',
            xaxis_title='Fold',
            yaxis_title='ROC AUC Score'
        )
        plots['learning_curves'] = fig_learning
        
        return plots
    
    def compare_feature_importance(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Compare feature importance across models"""
        importance_scores = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_scores[name] = model.feature_importances_
        
        return importance_scores
    
    def create_feature_importance_plots(
        self,
        importance_scores: Dict[str, np.ndarray]
    ) -> Dict[str, go.Figure]:
        """Create plots for feature importance comparison"""
        plots = {}
        
        for name, scores in importance_scores.items():
            fig = go.Figure()
            
            # Sort features by importance
            sorted_idx = np.argsort(scores)
            pos = np.arange(sorted_idx.shape[0]) + .5
            
            fig.add_trace(go.Bar(
                y=pos,
                x=scores[sorted_idx],
                orientation='h'
            ))
            
            fig.update_layout(
                title=f'Feature Importance - {name}',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=800
            )
            
            plots[name] = fig
        
        return plots
    
    def compare_dimensionality_reduction(
        self,
        X: np.ndarray,
        n_components: int = 2
    ) -> Dict[str, np.ndarray]:
        """Compare different dimensionality reduction techniques"""
        results = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components)
        results['PCA'] = pca.fit_transform(X_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=n_components, random_state=42)
        results['t-SNE'] = tsne.fit_transform(X_scaled)
        
        # UMAP
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        results['UMAP'] = reducer.fit_transform(X_scaled)
        
        return results
    
    def create_dimensionality_reduction_plots(
        self,
        results: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, go.Figure]:
        """Create plots for dimensionality reduction results"""
        plots = {}
        
        for name, embedding in results.items():
            fig = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                color=labels if labels is not None else None,
                title=f'{name} Projection'
            )
            
            fig.update_layout(
                xaxis_title=f'{name} Component 1',
                yaxis_title=f'{name} Component 2'
            )
            
            plots[name] = fig
        
        return plots
    
    def compare_clustering(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different clustering algorithms"""
        results = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        results['K-means'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X_scaled, kmeans_labels)
        }
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        if len(np.unique(dbscan_labels)) > 1:  # Only calculate if more than one cluster
            results['DBSCAN'] = {
                'labels': dbscan_labels,
                'silhouette': silhouette_score(X_scaled, dbscan_labels)
            }
        
        return results
    
    def create_clustering_plots(
        self,
        X: np.ndarray,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, go.Figure]:
        """Create plots for clustering results"""
        plots = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        for name, result in results.items():
            fig = px.scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                color=result['labels'],
                title=f'{name} Clustering'
            )
            
            fig.update_layout(
                xaxis_title='First Principal Component',
                yaxis_title='Second Principal Component'
            )
            
            plots[name] = fig
        
        return plots
    
    def explain_predictions(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate model explanations using SHAP and LIME"""
        explanations = {}
        
        # Use Random Forest for explanations
        model = self.models['Random Forest']
        
        # SHAP explanations
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        explanations['shap_values'] = shap_values
        
        # LIME explanation for first instance
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=['Benign', 'Malignant'],
            mode='classification'
        )
        
        lime_exp = lime_explainer.explain_instance(
            X[0],
            model.predict_proba,
            num_features=10
        )
        explanations['lime_explanation'] = lime_exp
        
        return explanations
    
    def create_explanation_plots(
        self,
        explanations: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Create plots for model explanations"""
        plots = {}
        
        # SHAP summary plot
        shap_values = explanations['shap_values']
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': range(len(feature_importance)),
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=feature_importance['feature'],
            x=feature_importance['importance'],
            orientation='h'
        ))
        
        fig_shap.update_layout(
            title='SHAP Feature Importance',
            xaxis_title='mean(|SHAP value|)',
            yaxis_title='Feature'
        )
        
        plots['shap_summary'] = fig_shap
        
        return plots
