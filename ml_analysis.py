import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr, kendalltau
import umap
import shap
import lime
import lime.lime_tabular
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class MLAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
    def compare_models(self, X, y, models=None):
        """Compare multiple models using cross-validation"""
        if models is None:
            models = self.models
            
        results = {}
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            # Learning curves
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            results[name] = {
                'cv_scores': cv_scores,
                'learning_curves': {
                    'train_sizes': train_sizes,
                    'train_scores': train_scores,
                    'test_scores': test_scores
                }
            }
            
        return results
    
    def compare_feature_importance(self, X, feature_names, models=None):
        """Compare feature importance across multiple models"""
        if models is None:
            models = self.models
            
        importance_scores = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                model.fit(X, np.zeros(len(X)))  # Dummy fit for feature importance
                importance_scores[name] = {
                    'scores': model.feature_importances_,
                    'features': feature_names
                }
                
        return importance_scores
    
    def compare_dimensionality_reduction(self, X, n_components=2):
        """Compare different dimensionality reduction techniques"""
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)
        results['pca'] = {
            'embedding': pca_result,
            'explained_variance': pca.explained_variance_ratio_
        }
        
        # Kernel PCA
        kpca = KernelPCA(n_components=n_components, kernel='rbf')
        kpca_result = kpca.fit_transform(X_scaled)
        results['kpca'] = {
            'embedding': kpca_result
        }
        
        # t-SNE
        tsne = TSNE(n_components=n_components, random_state=42)
        tsne_result = tsne.fit_transform(X_scaled)
        results['tsne'] = {
            'embedding': tsne_result
        }
        
        # UMAP
        umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
        umap_result = umap_reducer.fit_transform(X_scaled)
        results['umap'] = {
            'embedding': umap_result
        }
        
        # Isomap
        isomap = Isomap(n_components=n_components)
        isomap_result = isomap.fit_transform(X_scaled)
        results['isomap'] = {
            'embedding': isomap_result
        }
        
        return results
    
    def compare_clustering(self, X, n_clusters=3):
        """Compare different clustering algorithms"""
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X_scaled, kmeans_labels)
        }
        
        # DBSCAN
        dbscan = DBSCAN()
        dbscan_labels = dbscan.fit_predict(X_scaled)
        if len(np.unique(dbscan_labels)) > 1:
            results['dbscan'] = {
                'labels': dbscan_labels,
                'silhouette': silhouette_score(X_scaled, dbscan_labels)
            }
        
        # Hierarchical Clustering
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        hc_labels = hc.fit_predict(X_scaled)
        results['hierarchical'] = {
            'labels': hc_labels,
            'silhouette': silhouette_score(X_scaled, hc_labels),
            'linkage': linkage(X_scaled, method='ward')
        }
        
        return results
    
    def create_model_comparison_plots(self, results):
        """Create visualizations for model comparison"""
        plots = {}
        
        # ROC curves comparison
        fig = go.Figure()
        for name, result in results.items():
            fig.add_trace(go.Scatter(
                x=result['fpr'],
                y=result['tpr'],
                name=f"{name} (AUC = {result['auc']:.3f})"
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(dash='dash'),
            name='Random'
        ))
        
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        plots['roc_curves'] = fig
        
        # Learning curves comparison
        fig = make_subplots(
            rows=len(results), cols=1,
            subplot_titles=[name for name in results.keys()]
        )
        
        for i, (name, result) in enumerate(results.items(), 1):
            curves = result['learning_curves']
            
            fig.add_trace(
                go.Scatter(
                    x=curves['train_sizes'],
                    y=curves['train_scores'].mean(axis=1),
                    name=f"{name} (Train)",
                    line=dict(dash='solid')
                ),
                row=i, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=curves['train_sizes'],
                    y=curves['test_scores'].mean(axis=1),
                    name=f"{name} (Test)",
                    line=dict(dash='dash')
                ),
                row=i, col=1
            )
            
        fig.update_layout(
            height=300 * len(results),
            title="Learning Curves Comparison"
        )
        plots['learning_curves'] = fig
        
        return plots
    
    def create_clustering_plots(self, X, results):
        """Create visualizations for clustering comparison"""
        plots = {}
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.scaler.fit_transform(X))
        
        # Clustering results
        for method, result in results.items():
            if method != 'hierarchical':
                fig = px.scatter(
                    x=X_2d[:, 0], y=X_2d[:, 1],
                    color=result['labels'],
                    title=f"{method} Clustering Results"
                )
                plots[f'{method}_scatter'] = fig
        
        # Dendrogram for hierarchical clustering
        if 'hierarchical' in results:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['hierarchical']['linkage'][:, 2],
                y=np.arange(len(results['hierarchical']['linkage'])),
                mode='lines'
            ))
            fig.update_layout(title="Hierarchical Clustering Dendrogram")
            plots['dendrogram'] = fig
        
        return plots
    
    def create_dimensionality_reduction_plots(self, results, labels=None):
        """Create visualizations for dimensionality reduction comparison"""
        plots = {}
        
        for method, result in results.items():
            embedding = result['embedding']
            
            fig = px.scatter(
                x=embedding[:, 0], y=embedding[:, 1],
                color=labels if labels is not None else None,
                title=f"{method.upper()} Projection"
            )
            
            if 'explained_variance' in result:
                fig.update_layout(
                    annotations=[
                        dict(
                            text=f"Explained variance: {sum(result['explained_variance']):.2%}",
                            showarrow=False,
                            x=0.5, y=1.05
                        )
                    ]
                )
                
            plots[method] = fig
            
        return plots
    
    def create_feature_importance_plots(self, importance_scores):
        """Create visualizations for feature importance comparison"""
        plots = {}
        
        # Bar plots for each model
        for model_name, scores in importance_scores.items():
            fig = px.bar(
                x=scores['features'],
                y=scores['scores'],
                title=f"Feature Importance ({model_name})"
            )
            plots[f'{model_name}_importance'] = fig
        
        # Heatmap of feature importance correlation
        importance_matrix = np.array([
            scores['scores'] for scores in importance_scores.values()
        ])
        
        correlation_matrix = np.corrcoef(importance_matrix)
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(x="Model", y="Model"),
            x=list(importance_scores.keys()),
            y=list(importance_scores.keys()),
            title="Feature Importance Correlation Between Models"
        )
        plots['importance_correlation'] = fig
        
        return plots
    
    def explain_predictions(self, X, feature_names, models=None):
        """Generate model explanations using SHAP and LIME"""
        if models is None:
            models = self.models
            
        explanations = {}
        
        for name, model in models.items():
            # SHAP explanations
            explainer = shap.KernelExplainer(model.predict_proba, X)
            shap_values = explainer.shap_values(X)
            
            explanations[name] = {
                'shap_values': shap_values,
                'feature_names': feature_names
            }
            
            # LIME explanation for a sample
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                class_names=['0', '1'],
                mode='classification'
            )
            
            lime_exp = lime_explainer.explain_instance(
                X[0], model.predict_proba
            )
            
            explanations[name]['lime'] = lime_exp
            
        return explanations
    
    def create_explanation_plots(self, explanations):
        """Create visualizations for model explanations"""
        plots = {}
        
        for model_name, explanation in explanations.items():
            # SHAP summary plot
            fig = go.Figure()
            
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            
            # Sort features by importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_order = np.argsort(feature_importance)
            
            fig.add_trace(go.Box(
                x=[feature_names[i] for i in feature_order],
                y=[shap_values[:, i] for i in feature_order],
                name='SHAP values'
            ))
            
            fig.update_layout(
                title=f"SHAP Summary Plot ({model_name})",
                xaxis_title="Features",
                yaxis_title="SHAP value"
            )
            
            plots[f'{model_name}_shap'] = fig
            
            # LIME explanation plot
            lime_exp = explanation['lime']
            feature_weights = lime_exp.as_list()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[w for _, w in feature_weights],
                y=[f for f, _ in feature_weights],
                orientation='h'
            ))
            
            fig.update_layout(
                title=f"LIME Explanation ({model_name})",
                xaxis_title="Feature Weight",
                yaxis_title="Feature"
            )
            
            plots[f'{model_name}_lime'] = fig
            
        return plots
