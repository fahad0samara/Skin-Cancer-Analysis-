import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import torch
from skin_cancer_model import SkinCancerModel
from data_analyzer import DataAnalyzer
from advanced_analytics import AdvancedAnalytics
from batch_analyzer import BatchAnalyzer
from comparative_analysis import ComparativeAnalysis
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
import io
import tempfile
import shutil
from statistical_analysis import StatisticalAnalysis
from ml_analysis import MLAnalysis

class DashboardApp:
    def __init__(self):
        st.set_page_config(page_title="Skin Cancer Analysis Dashboard", layout="wide")
        self.model = SkinCancerModel()
        self.analyzer = DataAnalyzer()
        self.advanced = AdvancedAnalytics()
        self.batch_analyzer = BatchAnalyzer()
        self.comparative = ComparativeAnalysis()
        self.stats = StatisticalAnalysis()
        self.ml = MLAnalysis()
        
    def run(self):
        st.title("Skin Cancer Analysis Dashboard")
        
        # Sidebar for navigation
        page = st.sidebar.selectbox(
            "Select a Page",
            ["Data Overview", "Model Training", "Model Analysis", 
             "Predictions", "ABCD Analysis", "Advanced Analytics",
             "Feature Analysis", "Batch Analysis", "Comparative Analysis",
             "ML Analysis", "Export Reports"]
        )
        
        # Add session state for storing analysis results
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
            
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
            
        if 'comparative_results' not in st.session_state:
            st.session_state.comparative_results = []
        
        if page == "Data Overview":
            self.show_data_overview()
        elif page == "Model Training":
            self.show_model_training()
        elif page == "Model Analysis":
            self.show_model_analysis()
        elif page == "Predictions":
            self.show_predictions()
        elif page == "ABCD Analysis":
            self.show_abcd_analysis()
        elif page == "Advanced Analytics":
            self.show_advanced_analytics()
        elif page == "Feature Analysis":
            self.show_feature_analysis()
        elif page == "Batch Analysis":
            self.show_batch_analysis()
        elif page == "Comparative Analysis":
            self.show_comparative_analysis()
        elif page == "ML Analysis":
            self.show_ml_analysis()
        else:
            self.show_export_page()

    def show_data_overview(self):
        st.header("Dataset Overview")
        
        # Get dataset statistics
        stats = self.analyzer.get_dataset_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Images", stats['total_images'])
        with col2:
            st.metric("Benign Cases", stats['benign_count'])
        with col3:
            st.metric("Malignant Cases", stats['malignant_count'])
        
        # Class distribution plot
        st.subheader("Class Distribution")
        fig = px.pie(
            values=[stats['benign_count'], stats['malignant_count']],
            names=['Benign', 'Malignant'],
            title="Distribution of Classes"
        )
        st.plotly_chart(fig)
        
        # Image size distribution
        st.subheader("Image Size Distribution")
        fig = px.histogram(
            stats['image_sizes'],
            title="Distribution of Image Sizes"
        )
        st.plotly_chart(fig)
        
        # Sample images
        st.subheader("Sample Images")
        sample_images = self.analyzer.get_sample_images(n=5)
        cols = st.columns(5)
        for idx, (img, label) in enumerate(sample_images):
            cols[idx].image(img, caption=f"Class: {label}")

    def show_model_training(self):
        st.header("Model Training")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Number of Epochs", min_value=1, value=10)
            batch_size = st.number_input("Batch Size", min_value=1, value=32)
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%f")
            augmentation = st.checkbox("Use Data Augmentation", value=True)
        
        if st.button("Start Training"):
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training loop
            for epoch in range(epochs):
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                # Train epoch
                metrics = self.model.train_epoch(epoch, augmentation)
                
                # Update status
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {metrics['loss']:.4f} - Accuracy: {metrics['accuracy']:.2f}%")
                
                # Plot metrics
                self.plot_training_metrics(metrics)

    def show_model_analysis(self):
        st.header("Model Performance Analysis")
        
        # Model metrics
        metrics = self.analyzer.get_model_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{metrics['accuracy']:.2f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2f}%")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig = px.imshow(
            metrics['confusion_matrix'],
            labels=dict(x="Predicted", y="Actual"),
            x=['Benign', 'Malignant'],
            y=['Benign', 'Malignant']
        )
        st.plotly_chart(fig)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fig = px.line(
            x=metrics['fpr'],
            y=metrics['tpr'],
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            title=f"ROC Curve (AUC = {metrics['auc']:.3f})"
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig)

    def show_predictions(self):
        st.header("Make Predictions")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image")
            
            with col2:
                if st.button("Analyze"):
                    results = self.model.predict_image(image)
                    
                    # Display prediction
                    prediction = "Malignant" if results['prediction'] == 1 else "Benign"
                    confidence = max(results['probabilities']) * 100
                    
                    st.metric("Prediction", prediction)
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # ABCD metrics
                    st.subheader("ABCD Analysis")
                    for key, value in results['abcd_results'].items():
                        st.metric(key.capitalize(), f"{value:.3f}")

    def show_abcd_analysis(self):
        st.header("ABCD Criteria Analysis")
        
        # Get ABCD statistics
        abcd_stats = self.analyzer.get_abcd_statistics()
        
        # ABCD distribution plots
        st.subheader("ABCD Criteria Distribution")
        
        for criterion in ['asymmetry', 'border', 'color', 'diameter']:
            fig = px.histogram(
                abcd_stats[criterion],
                title=f"{criterion.capitalize()} Distribution",
                nbins=20
            )
            st.plotly_chart(fig)
        
        # Correlation matrix
        st.subheader("ABCD Criteria Correlation")
        correlation_matrix = self.analyzer.get_abcd_correlation()
        fig = px.imshow(
            correlation_matrix,
            labels=dict(x="Criteria", y="Criteria"),
            x=['Asymmetry', 'Border', 'Color', 'Diameter'],
            y=['Asymmetry', 'Border', 'Color', 'Diameter']
        )
        st.plotly_chart(fig)

    def show_advanced_analytics(self):
        st.header("Advanced Analytics")
        
        uploaded_file = st.file_uploader("Choose an image for analysis...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image")
                
                if st.button("Perform Advanced Analysis"):
                    # Perform segmentation
                    seg_results = self.advanced.image_segmentation(image)
                    
                    # Color analysis
                    color_stats = self.advanced.color_analysis(image, seg_results['mask'])
                    
                    # Texture analysis
                    texture_features = self.advanced.texture_analysis(image, seg_results['mask'])
                    
                    # Store results in session state
                    analysis_result = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'color_analysis': color_stats,
                        'texture_analysis': texture_features,
                        'shape_analysis': {
                            'area': seg_results['area'],
                            'perimeter': seg_results['perimeter']
                        }
                    }
                    st.session_state.analysis_results.append(analysis_result)
            
            with col2:
                # Show segmentation result
                st.image(seg_results['mask'], caption="Segmentation Mask")
                
                # Display metrics
                st.subheader("Shape Metrics")
                col3, col4 = st.columns(2)
                col3.metric("Area", f"{seg_results['area']:.2f}")
                col4.metric("Perimeter", f"{seg_results['perimeter']:.2f}")
                
                # Color analysis visualization
                st.subheader("Color Analysis")
                color_fig = go.Figure()
                for channel, stats in color_stats.items():
                    color_fig.add_trace(go.Bar(
                        name=channel,
                        x=['Mean', 'Std', 'Skewness', 'Kurtosis'],
                        y=[stats['mean'], stats['std'], stats['skewness'], stats['kurtosis']]
                    ))
                st.plotly_chart(color_fig)
                
                # Texture analysis visualization
                st.subheader("Texture Analysis")
                texture_fig = px.bar(
                    x=list(texture_features.keys()),
                    y=list(texture_features.values()),
                    title="Texture Features"
                )
                st.plotly_chart(texture_fig)

    def show_feature_analysis(self):
        st.header("Feature Analysis")
        
        # Get feature data
        features = self.analyzer.get_feature_data()
        
        # Dimension reduction
        st.subheader("Feature Space Visualization")
        method = st.selectbox("Select Dimension Reduction Method", ['t-SNE', 'PCA'])
        
        reduced_features = self.advanced.dimension_reduction(
            features, 
            method='tsne' if method == 't-SNE' else 'pca'
        )
        
        # Plot reduced features
        fig = px.scatter(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            color=self.analyzer.get_labels(),
            title=f"{method} Visualization"
        )
        st.plotly_chart(fig)
        
        # Cluster analysis
        st.subheader("Cluster Analysis")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        cluster_results = self.advanced.cluster_analysis(features, n_clusters)
        
        # Plot clusters
        fig = px.scatter(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            color=cluster_results['clusters'],
            title=f"Cluster Analysis (Silhouette Score: {cluster_results['silhouette_score']:.3f})"
        )
        st.plotly_chart(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_results = self.advanced.feature_importance_analysis(
            features,
            self.analyzer.get_labels()
        )
        
        fig = px.bar(
            x=self.analyzer.get_feature_names(),
            y=importance_results['importance'],
            title="Feature Importance Scores"
        )
        st.plotly_chart(fig)

    def show_batch_analysis(self):
        st.header("Batch Image Analysis")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images for analysis",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Images"):
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files to temp directory
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                    
                    # Process images
                    with st.spinner("Processing images..."):
                        results = self.batch_analyzer.batch_process(temp_dir)
                        st.session_state.batch_results = results
                
                # Show summary statistics
                summary = self.batch_analyzer.generate_summary_stats(results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", summary['total_images'])
                with col2:
                    st.metric("Successful", summary['successful'])
                with col3:
                    st.metric("Failed", summary['failed'])
                
                # Show visualizations
                st.subheader("Analysis Visualizations")
                visualizations = self.batch_analyzer.create_visualization_dashboard(results)
                
                # Display visualizations in tabs
                tabs = st.tabs([
                    "Predictions", "Confidence", "Correlations", "Scatter Matrix"
                ])
                
                with tabs[0]:
                    st.plotly_chart(visualizations['prediction_dist'])
                
                with tabs[1]:
                    st.plotly_chart(visualizations['confidence_hist'])
                
                with tabs[2]:
                    st.plotly_chart(visualizations['correlation_matrix'])
                
                with tabs[3]:
                    st.plotly_chart(visualizations['scatter_matrix'])
                
                # Export options
                st.subheader("Export Results")
                if st.button("Export to Excel"):
                    # Create temporary file for Excel
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.xlsx'
                    ) as tmp_file:
                        self.batch_analyzer.export_results(results, tmp_file.name)
                        
                        with open(tmp_file.name, 'rb') as f:
                            excel_data = f.read()
                        
                        st.download_button(
                            "Download Excel Report",
                            excel_data,
                            "batch_analysis_report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Clean up
                        os.unlink(tmp_file.name)

    def show_comparative_analysis(self):
        st.header("Comparative Analysis")
        
        # Add tabs for different types of comparison
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dataset Comparison",
            "Model Comparison",
            "Time Series Analysis",
            "Advanced Statistics"
        ])
        
        with tab1:
            st.subheader("Dataset Comparison")
            
            # File upload for two datasets
            col1, col2 = st.columns(2)
            
            with col1:
                dataset1 = st.file_uploader("Upload first dataset", key="dataset1")
                label1 = st.text_input("Label for first dataset", "Dataset 1")
                
            with col2:
                dataset2 = st.file_uploader("Upload second dataset", key="dataset2")
                label2 = st.text_input("Label for second dataset", "Dataset 2")
            
            if dataset1 is not None and dataset2 is not None:
                if st.button("Compare Datasets"):
                    # Load and process datasets
                    data1 = pd.read_csv(dataset1)
                    data2 = pd.read_csv(dataset2)
                    
                    # Statistical tests
                    st.subheader("Statistical Tests")
                    test_results = self.stats.perform_statistical_tests(
                        data1.values.flatten(),
                        data2.values.flatten()
                    )
                    
                    # Display test results in expandable sections
                    with st.expander("Normality Tests"):
                        st.json(test_results['normality'])
                        
                    with st.expander("Variance Homogeneity"):
                        st.json(test_results['variance_homogeneity'])
                        
                    with st.expander("Difference Tests"):
                        st.json(test_results['difference_tests'])
                        
                    with st.expander("Effect Size"):
                        st.json(test_results['effect_size'])
                    
                    # Advanced visualizations
                    st.subheader("Advanced Visualizations")
                    visualizations = self.stats.create_advanced_visualizations(
                        data1.values.flatten(),
                        data2.values.flatten(),
                        labels=[label1, label2]
                    )
                    
                    viz_tabs = st.tabs([
                        "Q-Q Plot", "ECDF", "PCA", "t-SNE"
                    ])
                    
                    with viz_tabs[0]:
                        st.plotly_chart(visualizations['qq_plot'])
                    
                    with viz_tabs[1]:
                        st.plotly_chart(visualizations['ecdf_plot'])
                    
                    with viz_tabs[2]:
                        st.plotly_chart(visualizations['pca_plot'])
                    
                    with viz_tabs[3]:
                        st.plotly_chart(visualizations['tsne_plot'])
                    
                    # Export options
                    st.subheader("Export Results")
                    export_format = st.selectbox(
                        "Select export format",
                        ["JSON", "CSV", "Excel", "HTML"]
                    )
                    
                    if st.button("Export Analysis"):
                        results = {
                            'statistical_tests': test_results,
                            'basic_statistics': {
                                label1: data1.describe().to_dict(),
                                label2: data2.describe().to_dict()
                            }
                        }
                        
                        export_data = self.stats.export_analysis(
                            results,
                            format=export_format.lower()
                        )
                        
                        if export_format == "Excel":
                            with open(export_data, 'rb') as f:
                                st.download_button(
                                    "Download Analysis",
                                    f,
                                    "analysis_results.xlsx"
                                )
                        else:
                            st.download_button(
                                "Download Analysis",
                                export_data,
                                f"analysis_results.{export_format.lower()}"
                            )
        
        with tab4:
            st.subheader("Advanced Statistical Analysis")
            
            # Multiple dataset comparison
            st.write("Compare Multiple Datasets")
            
            num_datasets = st.number_input(
                "Number of datasets to compare",
                min_value=2,
                max_value=10,
                value=3
            )
            
            datasets = []
            labels = []
            
            cols = st.columns(num_datasets)
            for i, col in enumerate(cols):
                with col:
                    dataset = st.file_uploader(
                        f"Upload dataset {i+1}",
                        key=f"advanced_dataset_{i}"
                    )
                    label = st.text_input(
                        f"Label for dataset {i+1}",
                        f"Dataset {i+1}",
                        key=f"advanced_label_{i}"
                    )
                    if dataset is not None:
                        data = pd.read_csv(dataset)
                        datasets.append(data.values.flatten())
                        labels.append(label)
            
            if len(datasets) == num_datasets:
                if st.button("Perform ANOVA"):
                    # Perform ANOVA
                    anova_results = self.stats.perform_anova(
                        *datasets,
                        labels=labels
                    )
                    
                    # Display ANOVA results
                    st.write("ANOVA Results")
                    st.json(anova_results['anova'])
                    
                    # Display Tukey's HSD results
                    st.write("Tukey's HSD Test Results")
                    st.write(anova_results['tukey_hsd'])
                    
                    # Display effect size
                    st.write("Effect Size")
                    st.json(anova_results['effect_size'])
                    
                    # Create visualizations
                    fig = go.Figure()
                    
                    for data, label in zip(datasets, labels):
                        fig.add_trace(go.Box(
                            y=data,
                            name=label
                        ))
                    
                    fig.update_layout(
                        title="Distribution Comparison",
                        yaxis_title="Value"
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Export options
                    if st.button("Export ANOVA Results"):
                        export_data = pd.DataFrame({
                            'ANOVA': [anova_results['anova']],
                            'Effect Size': [anova_results['effect_size']]
                        }).to_csv(index=False)
                        
                        st.download_button(
                            "Download ANOVA Results",
                            export_data,
                            "anova_results.csv"
                        )
                    
    def show_ml_analysis(self):
        st.header("Machine Learning Analysis")
        
        # Create tabs for different ML analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Model Comparison",
            "Feature Analysis",
            "Dimensionality Reduction",
            "Clustering Analysis",
            "Model Explanations"
        ])
        
        with tab1:
            st.subheader("Model Comparison")
            
            # File upload for training data
            train_data = st.file_uploader(
                "Upload training data (CSV)",
                key="train_data"
            )
            
            if train_data is not None:
                data = pd.read_csv(train_data)
                
                # Select target variable
                target_col = st.selectbox(
                    "Select target variable",
                    data.columns
                )
                
                # Select features
                feature_cols = st.multiselect(
                    "Select features",
                    [col for col in data.columns if col != target_col],
                    default=[col for col in data.columns if col != target_col]
                )
                
                if st.button("Compare Models"):
                    X = data[feature_cols].values
                    y = data[target_col].values
                    
                    with st.spinner("Comparing models..."):
                        # Compare models
                        results = self.ml.compare_models(X, y)
                        
                        # Create and display plots
                        plots = self.ml.create_model_comparison_plots(results)
                        
                        st.plotly_chart(plots['roc_curves'])
                        st.plotly_chart(plots['learning_curves'])
                        
                        # Display cross-validation results
                        st.subheader("Cross-validation Results")
                        cv_results = {
                            name: {
                                'Mean ROC-AUC': result['cv_scores'].mean(),
                                'Std ROC-AUC': result['cv_scores'].std()
                            }
                            for name, result in results.items()
                        }
                        st.dataframe(pd.DataFrame(cv_results).T)
        
        with tab2:
            st.subheader("Feature Analysis")
            
            if train_data is not None and 'feature_cols' in locals():
                if st.button("Analyze Features"):
                    with st.spinner("Analyzing features..."):
                        # Compare feature importance
                        importance_scores = self.ml.compare_feature_importance(
                            X, feature_cols
                        )
                        
                        # Create and display plots
                        plots = self.ml.create_feature_importance_plots(
                            importance_scores
                        )
                        
                        for name, plot in plots.items():
                            st.plotly_chart(plot)
        
        with tab3:
            st.subheader("Dimensionality Reduction")
            
            if train_data is not None and 'feature_cols' in locals():
                n_components = st.slider(
                    "Number of components",
                    min_value=2,
                    max_value=min(5, len(feature_cols)),
                    value=2
                )
                
                if st.button("Reduce Dimensionality"):
                    with st.spinner("Reducing dimensionality..."):
                        # Compare dimensionality reduction
                        results = self.ml.compare_dimensionality_reduction(
                            X, n_components
                        )
                        
                        # Create and display plots
                        plots = self.ml.create_dimensionality_reduction_plots(
                            results,
                            labels=y if 'y' in locals() else None
                        )
                        
                        for name, plot in plots.items():
                            st.plotly_chart(plot)
        
        with tab4:
            st.subheader("Clustering Analysis")
            
            if train_data is not None and 'feature_cols' in locals():
                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=3
                )
                
                if st.button("Perform Clustering"):
                    with st.spinner("Clustering data..."):
                        # Compare clustering algorithms
                        results = self.ml.compare_clustering(X, n_clusters)
                        
                        # Create and display plots
                        plots = self.ml.create_clustering_plots(X, results)
                        
                        for name, plot in plots.items():
                            st.plotly_chart(plot)
                        
                        # Display silhouette scores
                        st.subheader("Clustering Performance")
                        silhouette_scores = {
                            name: result['silhouette']
                            for name, result in results.items()
                            if 'silhouette' in result
                        }
                        st.dataframe(pd.DataFrame(
                            silhouette_scores.items(),
                            columns=['Method', 'Silhouette Score']
                        ))
        
        with tab5:
            st.subheader("Model Explanations")
            
            if train_data is not None and 'feature_cols' in locals():
                if st.button("Generate Explanations"):
                    with st.spinner("Generating model explanations..."):
                        # Generate explanations
                        explanations = self.ml.explain_predictions(
                            X, feature_cols
                        )
                        
                        # Create and display plots
                        plots = self.ml.create_explanation_plots(explanations)
                        
                        for name, plot in plots.items():
                            st.plotly_chart(plot)
                            
                        # Export options
                        if st.button("Export ML Analysis"):
                            # Prepare results for export
                            export_data = {
                                'model_comparison': results if 'results' in locals() else None,
                                'feature_importance': importance_scores if 'importance_scores' in locals() else None,
                                'clustering': results if 'results' in locals() else None
                            }
                            
                            # Convert to DataFrame
                            export_df = pd.json_normalize(export_data)
                            
                            # Download button
                            st.download_button(
                                "Download Analysis Results",
                                export_df.to_csv(index=False),
                                "ml_analysis_results.csv",
                                "text/csv"
                            )
                            
    def show_export_page(self):
        st.header("Export Analysis Reports")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis results available. Please perform some analyses first.")
            return
        
        # Display available results
        st.subheader("Available Analysis Results")
        for i, result in enumerate(st.session_state.analysis_results):
            st.write(f"{i+1}. {result['filename']} - {result['timestamp']}")
        
        # Export options
        st.subheader("Export Options")
        export_format = st.selectbox("Select Export Format", ["Excel", "JSON", "CSV"])
        
        if st.button("Export Results"):
            if export_format == "Excel":
                # Create Excel file
                output = self.create_excel_report(st.session_state.analysis_results)
                st.download_button(
                    "Download Excel Report",
                    output,
                    "analysis_report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                # Create JSON file
                output = json.dumps(st.session_state.analysis_results, indent=2)
                st.download_button(
                    "Download JSON Report",
                    output,
                    "analysis_report.json",
                    "application/json"
                )
            else:
                # Create CSV file
                output = self.create_csv_report(st.session_state.analysis_results)
                st.download_button(
                    "Download CSV Report",
                    output,
                    "analysis_report.csv",
                    "text/csv"
                )

    def create_excel_report(self, results):
        """Create Excel report from analysis results"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = []
            for result in results:
                summary_data.append({
                    'Timestamp': result['timestamp'],
                    'Filename': result['filename'],
                    'Area': result['shape_analysis']['area'],
                    'Perimeter': result['shape_analysis']['perimeter']
                })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed sheets for each analysis
            for result in results:
                # Color analysis
                pd.DataFrame(result['color_analysis']).to_excel(
                    writer, 
                    sheet_name=f"{result['filename'][:10]}_color"
                )
                
                # Texture analysis
                pd.DataFrame([result['texture_analysis']]).to_excel(
                    writer,
                    sheet_name=f"{result['filename'][:10]}_texture"
                )
        
        return output.getvalue()

    def create_csv_report(self, results):
        """Create CSV report from analysis results"""
        # Flatten the results into a DataFrame
        rows = []
        for result in results:
            row = {
                'Timestamp': result['timestamp'],
                'Filename': result['filename'],
                'Area': result['shape_analysis']['area'],
                'Perimeter': result['shape_analysis']['perimeter']
            }
            
            # Add color analysis
            for channel, stats in result['color_analysis'].items():
                for stat_name, value in stats.items():
                    row[f'{channel}_{stat_name}'] = value
            
            # Add texture analysis
            for feature_name, value in result['texture_analysis'].items():
                row[f'texture_{feature_name}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows).to_csv(index=False)

    def plot_training_metrics(self, metrics):
        # Create two columns for loss and accuracy plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                y=metrics['loss_history'],
                title="Training Loss",
                labels={'x': 'Epoch', 'y': 'Loss'}
            )
            st.plotly_chart(fig)
        
        with col2:
            fig = px.line(
                y=metrics['accuracy_history'],
                title="Training Accuracy",
                labels={'x': 'Epoch', 'y': 'Accuracy (%)'}
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    app = DashboardApp()
    app.run()
