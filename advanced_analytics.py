import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from torchvision import models
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

class AdvancedAnalytics:
    def __init__(self):
        self.feature_extractor = self._create_feature_extractor()
        
    def _create_feature_extractor(self):
        """Create a feature extractor using pretrained ResNet"""
        model = models.resnet50(pretrained=True)
        return nn.Sequential(*list(model.children())[:-1])
    
    def extract_features(self, image_tensor):
        """Extract features from image using the feature extractor"""
        with torch.no_grad():
            features = self.feature_extractor(image_tensor.unsqueeze(0))
            return features.squeeze().numpy()
    
    def dimension_reduction(self, features, method='tsne'):
        """Perform dimension reduction using t-SNE or PCA"""
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        reduced_features = reducer.fit_transform(features)
        return reduced_features
    
    def cluster_analysis(self, features, n_clusters=3):
        """Perform cluster analysis using K-means"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Calculate cluster metrics
        silhouette = sklearn.metrics.silhouette_score(features, clusters)
        
        return {
            'clusters': clusters,
            'centroids': kmeans.cluster_centers_,
            'silhouette_score': silhouette
        }
    
    def feature_importance_analysis(self, features, labels):
        """Analyze feature importance using Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        
        return {
            'importance': rf.feature_importances_,
            'cross_val_scores': cross_val_score(rf, features, labels, cv=5)
        }
    
    def image_segmentation(self, image):
        """Perform image segmentation for lesion analysis"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (assumed to be the lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)
        
        return {
            'mask': mask,
            'contour': largest_contour,
            'area': cv2.contourArea(largest_contour),
            'perimeter': cv2.arcLength(largest_contour, True)
        }
    
    def color_analysis(self, image, mask):
        """Analyze color distribution within the lesion"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply mask
        masked_img = cv2.bitwise_and(img_array, img_array, mask=mask)
        
        # Calculate color statistics for each channel
        color_stats = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_values = masked_img[:,:,i][mask > 0]
            color_stats[channel] = {
                'mean': np.mean(channel_values),
                'std': np.std(channel_values),
                'skewness': stats.skew(channel_values),
                'kurtosis': stats.kurtosis(channel_values)
            }
            
        return color_stats
    
    def texture_analysis(self, image, mask):
        """Analyze texture features within the lesion"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Calculate GLCM (Gray-Level Co-occurrence Matrix)
        glcm = skimage.feature.graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        
        # Calculate texture properties
        texture_features = {
            'contrast': skimage.feature.graycoprops(glcm, 'contrast')[0, 0],
            'dissimilarity': skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0],
            'homogeneity': skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': skimage.feature.graycoprops(glcm, 'energy')[0, 0],
            'correlation': skimage.feature.graycoprops(glcm, 'correlation')[0, 0]
        }
        
        return texture_features
    
    def generate_report(self, image_path):
        """Generate comprehensive analysis report for an image"""
        # Load image
        image = Image.open(image_path)
        
        # Perform segmentation
        seg_results = self.image_segmentation(image)
        
        # Color analysis
        color_stats = self.color_analysis(image, seg_results['mask'])
        
        # Texture analysis
        texture_features = self.texture_analysis(image, seg_results['mask'])
        
        # Shape analysis
        shape_metrics = {
            'area': seg_results['area'],
            'perimeter': seg_results['perimeter'],
            'circularity': 4 * np.pi * seg_results['area'] / (seg_results['perimeter'] ** 2),
            'aspect_ratio': cv2.minAreaRect(seg_results['contour'])[1][1] / 
                          cv2.minAreaRect(seg_results['contour'])[1][0]
        }
        
        return {
            'color_analysis': color_stats,
            'texture_analysis': texture_features,
            'shape_analysis': shape_metrics
        }
    
    def plot_feature_distribution(self, features, feature_names):
        """Create distribution plots for features"""
        fig = go.Figure()
        
        for i, name in enumerate(feature_names):
            fig.add_trace(go.Violin(
                y=features[:, i],
                name=name,
                box_visible=True,
                meanline_visible=True
            ))
            
        fig.update_layout(
            title="Feature Distributions",
            yaxis_title="Value",
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_matrix(self, features, feature_names):
        """Create correlation matrix heatmap"""
        corr_matrix = np.corrcoef(features.T)
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features"),
            x=feature_names,
            y=feature_names,
            title="Feature Correlation Matrix"
        )
        
        return fig
    
    def export_analysis(self, analysis_results, output_path):
        """Export analysis results to Excel file"""
        with pd.ExcelWriter(output_path) as writer:
            # Color analysis
            pd.DataFrame(analysis_results['color_analysis']).to_excel(writer, sheet_name='Color Analysis')
            
            # Texture analysis
            pd.DataFrame([analysis_results['texture_analysis']]).to_excel(writer, sheet_name='Texture Analysis')
            
            # Shape analysis
            pd.DataFrame([analysis_results['shape_analysis']]).to_excel(writer, sheet_name='Shape Analysis')
