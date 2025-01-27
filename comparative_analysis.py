import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class ComparativeAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def compare_datasets(self, dataset1, dataset2, labels=None):
        """Compare two datasets and generate statistical comparisons"""
        if labels is None:
            labels = ['Dataset 1', 'Dataset 2']
            
        comparisons = {}
        
        # Basic statistics
        for i, dataset in enumerate([dataset1, dataset2]):
            label = labels[i]
            comparisons[label] = {
                'mean': np.mean(dataset),
                'median': np.median(dataset),
                'std': np.std(dataset),
                'min': np.min(dataset),
                'max': np.max(dataset),
                'skewness': stats.skew(dataset),
                'kurtosis': stats.kurtosis(dataset)
            }
            
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(dataset1, dataset2)
        comparisons['statistical_tests'] = {
            't_statistic': t_stat,
            'p_value': p_value
        }
        
        return comparisons
    
    def create_comparative_visualizations(self, data1, data2, labels=None):
        """Create comparative visualizations between two datasets"""
        if labels is None:
            labels = ['Dataset 1', 'Dataset 2']
            
        visualizations = {}
        
        # Box plot comparison
        fig = go.Figure()
        fig.add_trace(go.Box(y=data1, name=labels[0]))
        fig.add_trace(go.Box(y=data2, name=labels[1]))
        fig.update_layout(title="Distribution Comparison")
        visualizations['box_plot'] = fig
        
        # Violin plot comparison
        fig = go.Figure()
        fig.add_trace(go.Violin(y=data1, name=labels[0], side='negative'))
        fig.add_trace(go.Violin(y=data2, name=labels[1], side='positive'))
        fig.update_layout(title="Density Comparison")
        visualizations['violin_plot'] = fig
        
        # Histogram comparison
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data1, name=labels[0], opacity=0.75))
        fig.add_trace(go.Histogram(x=data2, name=labels[1], opacity=0.75))
        fig.update_layout(barmode='overlay', title="Distribution Overlap")
        visualizations['histogram'] = fig
        
        return visualizations
    
    def compare_time_series(self, time_series1, time_series2, dates, labels=None):
        """Compare two time series datasets"""
        if labels is None:
            labels = ['Series 1', 'Series 2']
            
        # Create time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=time_series1, name=labels[0]))
        fig.add_trace(go.Scatter(x=dates, y=time_series2, name=labels[1]))
        fig.update_layout(title="Time Series Comparison")
        
        # Calculate rolling statistics
        window = min(len(time_series1) // 10, 30)  # Dynamic window size
        rolling_mean1 = pd.Series(time_series1).rolling(window=window).mean()
        rolling_mean2 = pd.Series(time_series2).rolling(window=window).mean()
        
        # Add rolling means
        fig.add_trace(go.Scatter(
            x=dates, y=rolling_mean1,
            name=f"{labels[0]} (Moving Avg)",
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=rolling_mean2,
            name=f"{labels[1]} (Moving Avg)",
            line=dict(dash='dash')
        ))
        
        return fig
    
    def compare_feature_importance(self, features1, features2, feature_names, labels=None):
        """Compare feature importance between two models or datasets"""
        if labels is None:
            labels = ['Model 1', 'Model 2']
            
        # Create comparison bar plot
        fig = go.Figure()
        
        # Add bars for both models
        x = np.arange(len(feature_names))
        width = 0.35
        
        fig.add_trace(go.Bar(
            x=x - width/2,
            y=features1,
            name=labels[0],
            width=width
        ))
        
        fig.add_trace(go.Bar(
            x=x + width/2,
            y=features2,
            name=labels[1],
            width=width
        ))
        
        fig.update_layout(
            title="Feature Importance Comparison",
            xaxis=dict(
                ticktext=feature_names,
                tickvals=x,
                title="Features"
            ),
            yaxis=dict(title="Importance"),
            barmode='group'
        )
        
        return fig
    
    def compare_roc_curves(self, y_true1, y_pred1, y_true2, y_pred2, labels=None):
        """Compare ROC curves between two models"""
        if labels is None:
            labels = ['Model 1', 'Model 2']
            
        # Calculate ROC curves
        fpr1, tpr1, _ = roc_curve(y_true1, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true2, y_pred2)
        
        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        
        # Create ROC plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr1, y=tpr1,
            name=f"{labels[0]} (AUC = {roc_auc1:.3f})"
        ))
        
        fig.add_trace(go.Scatter(
            x=fpr2, y=tpr2,
            name=f"{labels[1]} (AUC = {roc_auc2:.3f})"
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        ))
        
        fig.update_layout(
            title="ROC Curve Comparison",
            xaxis=dict(title="False Positive Rate"),
            yaxis=dict(title="True Positive Rate")
        )
        
        return fig
    
    def create_radar_comparison(self, metrics1, metrics2, metric_names, labels=None):
        """Create radar chart comparing metrics between two datasets/models"""
        if labels is None:
            labels = ['Dataset 1', 'Dataset 2']
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=metrics1,
            theta=metric_names,
            fill='toself',
            name=labels[0]
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=metrics2,
            theta=metric_names,
            fill='toself',
            name=labels[1]
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(metrics1), max(metrics2))]
                )
            ),
            title="Metric Comparison"
        )
        
        return fig
    
    def compare_correlation_matrices(self, corr1, corr2, labels=None):
        """Compare correlation matrices between two datasets"""
        if labels is None:
            labels = ['Dataset 1', 'Dataset 2']
            
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=labels
        )
        
        # Add heatmaps
        fig.add_trace(
            go.Heatmap(z=corr1, colorscale='RdBu', zmid=0),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(z=corr2, colorscale='RdBu', zmid=0),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Correlation Matrix Comparison",
            height=600
        )
        
        return fig
    
    def generate_comparison_report(self, results1, results2, labels=None):
        """Generate comprehensive comparison report"""
        if labels is None:
            labels = ['Dataset 1', 'Dataset 2']
            
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'labels': labels,
            'basic_statistics': {},
            'statistical_tests': {},
            'performance_metrics': {},
            'differences': {}
        }
        
        # Compare basic statistics
        for metric in ['mean', 'median', 'std', 'min', 'max']:
            report['basic_statistics'][metric] = {
                labels[0]: results1[metric],
                labels[1]: results2[metric],
                'difference': results1[metric] - results2[metric],
                'percent_change': ((results1[metric] - results2[metric]) / results2[metric]) * 100
            }
        
        # Statistical significance tests
        t_stat, p_value = stats.ttest_ind(results1['data'], results2['data'])
        report['statistical_tests'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return report
