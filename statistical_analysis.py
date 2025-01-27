import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import pingouin as pg

class StatisticalAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def perform_statistical_tests(self, data1, data2):
        """Perform comprehensive statistical tests between two datasets"""
        tests = {}
        
        # Normality tests
        _, shapiro_p1 = stats.shapiro(data1)
        _, shapiro_p2 = stats.shapiro(data2)
        tests['normality'] = {
            'dataset1_shapiro_p': shapiro_p1,
            'dataset2_shapiro_p': shapiro_p2,
            'is_normal': shapiro_p1 > 0.05 and shapiro_p2 > 0.05
        }
        
        # Homogeneity of variance
        _, levene_p = stats.levene(data1, data2)
        tests['variance_homogeneity'] = {
            'levene_p': levene_p,
            'is_homogeneous': levene_p > 0.05
        }
        
        # T-test and Mann-Whitney U test
        t_stat, t_p = stats.ttest_ind(data1, data2)
        u_stat, u_p = stats.mannwhitneyu(data1, data2)
        tests['difference_tests'] = {
            't_test': {'statistic': t_stat, 'p_value': t_p},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_p}
        }
        
        # Effect size calculations
        cohens_d = (np.mean(data1) - np.mean(data2)) / np.sqrt(
            (np.var(data1) + np.var(data2)) / 2
        )
        tests['effect_size'] = {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(data1, data2)
        tests['distribution_comparison'] = {
            'ks_test': {'statistic': ks_stat, 'p_value': ks_p}
        }
        
        return tests
    
    def perform_anova(self, *datasets, labels=None):
        """Perform one-way ANOVA with post-hoc tests"""
        if labels is None:
            labels = [f"Dataset {i+1}" for i in range(len(datasets))]
            
        # Prepare data for ANOVA
        all_data = []
        groups = []
        for i, data in enumerate(datasets):
            all_data.extend(data)
            groups.extend([labels[i]] * len(data))
            
        # One-way ANOVA
        f_stat, anova_p = stats.f_oneway(*datasets)
        
        # Tukey's HSD test
        tukey = pairwise_tukeyhsd(all_data, groups)
        
        # Effect size (Eta-squared)
        df_between = len(datasets) - 1
        df_within = len(all_data) - len(datasets)
        ss_between = sum(len(d) * (np.mean(d) - np.mean(all_data))**2 for d in datasets)
        ss_total = sum((x - np.mean(all_data))**2 for x in all_data)
        eta_squared = ss_between / ss_total
        
        return {
            'anova': {'f_statistic': f_stat, 'p_value': anova_p},
            'tukey_hsd': tukey,
            'effect_size': {'eta_squared': eta_squared}
        }
    
    def analyze_time_series(self, time_series, period=None):
        """Perform comprehensive time series analysis"""
        analysis = {}
        
        # Decomposition
        if period is None:
            period = self._estimate_period(time_series)
        
        decomposition = seasonal_decompose(
            time_series, period=period, extrapolate_trend='freq'
        )
        
        analysis['decomposition'] = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
        
        # Stationarity tests
        adf_result = adfuller(time_series)
        kpss_result = kpss(time_series)
        
        analysis['stationarity'] = {
            'adf_test': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            },
            'kpss_test': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
        
        # Autocorrelation analysis
        acf = sm.tsa.acf(time_series, nlags=40)
        pacf = sm.tsa.pacf(time_series, nlags=40)
        
        analysis['correlation'] = {
            'acf': acf,
            'pacf': pacf
        }
        
        return analysis
    
    def create_advanced_visualizations(self, data1, data2, labels=None):
        """Create advanced statistical visualizations"""
        if labels is None:
            labels = ['Dataset 1', 'Dataset 2']
            
        visualizations = {}
        
        # Q-Q plot
        fig = go.Figure()
        
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(data1)))
        
        for data, label in zip([data1, data2], labels):
            sorted_data = np.sort(data)
            fig.add_trace(go.Scatter(
                x=theoretical_q,
                y=sorted_data,
                mode='markers',
                name=label
            ))
            
        fig.add_trace(go.Scatter(
            x=theoretical_q,
            y=theoretical_q,
            mode='lines',
            name='Normal',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title="Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles"
        )
        
        visualizations['qq_plot'] = fig
        
        # ECDF plot
        fig = go.Figure()
        
        for data, label in zip([data1, data2], labels):
            sorted_data = np.sort(data)
            ecdf = np.arange(1, len(data) + 1) / len(data)
            
            fig.add_trace(go.Scatter(
                x=sorted_data,
                y=ecdf,
                mode='lines',
                name=label
            ))
            
        fig.update_layout(
            title="Empirical Cumulative Distribution Function",
            xaxis_title="Value",
            yaxis_title="ECDF"
        )
        
        visualizations['ecdf_plot'] = fig
        
        # 2D visualization using PCA
        combined_data = np.vstack([data1, data2])
        scaled_data = self.scaler.fit_transform(combined_data.reshape(-1, 1))
        
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=42)
        
        # PCA transformation
        pca_result = pca.fit_transform(scaled_data)
        
        fig = go.Figure()
        
        n1 = len(data1)
        fig.add_trace(go.Scatter(
            x=pca_result[:n1, 0],
            y=pca_result[:n1, 1],
            mode='markers',
            name=labels[0]
        ))
        
        fig.add_trace(go.Scatter(
            x=pca_result[n1:, 0],
            y=pca_result[n1:, 1],
            mode='markers',
            name=labels[1]
        ))
        
        fig.update_layout(
            title="PCA Visualization",
            xaxis_title="First Principal Component",
            yaxis_title="Second Principal Component"
        )
        
        visualizations['pca_plot'] = fig
        
        # t-SNE visualization
        tsne_result = tsne.fit_transform(scaled_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=tsne_result[:n1, 0],
            y=tsne_result[:n1, 1],
            mode='markers',
            name=labels[0]
        ))
        
        fig.add_trace(go.Scatter(
            x=tsne_result[n1:, 0],
            y=tsne_result[n1:, 1],
            mode='markers',
            name=labels[1]
        ))
        
        fig.update_layout(
            title="t-SNE Visualization",
            xaxis_title="First t-SNE Component",
            yaxis_title="Second t-SNE Component"
        )
        
        visualizations['tsne_plot'] = fig
        
        return visualizations
    
    def create_time_series_visualizations(self, analysis_results):
        """Create visualizations for time series analysis"""
        visualizations = {}
        
        # Decomposition plot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Original", "Trend", "Seasonal", "Residual"]
        )
        
        components = [
            analysis_results['decomposition']['trend'],
            analysis_results['decomposition']['seasonal'],
            analysis_results['decomposition']['residual']
        ]
        
        for i, component in enumerate(components, start=2):
            fig.add_trace(
                go.Scatter(y=component, mode='lines'),
                row=i, col=1
            )
            
        fig.update_layout(height=800, title="Time Series Decomposition")
        visualizations['decomposition'] = fig
        
        # ACF/PACF plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Autocorrelation", "Partial Autocorrelation"]
        )
        
        fig.add_trace(
            go.Bar(
                x=np.arange(len(analysis_results['correlation']['acf'])),
                y=analysis_results['correlation']['acf']
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=np.arange(len(analysis_results['correlation']['pacf'])),
                y=analysis_results['correlation']['pacf']
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title="Correlation Analysis")
        visualizations['correlation'] = fig
        
        return visualizations
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def _estimate_period(self, time_series):
        """Estimate the period of a time series using autocorrelation"""
        acf = sm.tsa.acf(time_series, nlags=len(time_series)//2)
        peaks = np.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
        if len(peaks) > 0:
            return peaks[0]
        return 1  # Default to 1 if no clear periodicity is found
    
    def export_analysis(self, analysis_results, format='json'):
        """Export analysis results in various formats"""
        if format == 'json':
            return pd.json_normalize(analysis_results).to_json(orient='records')
        elif format == 'csv':
            return pd.json_normalize(analysis_results).to_csv(index=False)
        elif format == 'excel':
            with pd.ExcelWriter('analysis_results.xlsx') as writer:
                for key, value in analysis_results.items():
                    pd.json_normalize(value).to_excel(writer, sheet_name=key)
            return 'analysis_results.xlsx'
        elif format == 'html':
            return pd.json_normalize(analysis_results).to_html()
        else:
            raise ValueError(f"Unsupported format: {format}")
