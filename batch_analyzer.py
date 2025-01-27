import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from advanced_analytics import AdvancedAnalytics
from skin_cancer_model import SkinCancerModel
import plotly.express as px
import plotly.graph_objects as go

class BatchAnalyzer:
    def __init__(self):
        self.advanced = AdvancedAnalytics()
        self.model = SkinCancerModel()
        
    def process_image(self, image_path):
        """Process a single image and return its analysis results"""
        try:
            image = Image.open(image_path)
            
            # Get basic image info
            img_info = {
                'filename': os.path.basename(image_path),
                'size': image.size,
                'mode': image.mode
            }
            
            # Perform segmentation
            seg_results = self.advanced.image_segmentation(image)
            
            # Color analysis
            color_stats = self.advanced.color_analysis(image, seg_results['mask'])
            
            # Texture analysis
            texture_features = self.advanced.texture_analysis(image, seg_results['mask'])
            
            # Model prediction
            prediction_results = self.model.predict_image(image_path)
            
            return {
                'image_info': img_info,
                'segmentation': seg_results,
                'color_analysis': color_stats,
                'texture_analysis': texture_features,
                'prediction': prediction_results
            }
            
        except Exception as e:
            return {
                'filename': os.path.basename(image_path),
                'error': str(e)
            }
    
    def batch_process(self, image_dir, max_workers=4):
        """Process all images in a directory"""
        image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(self.process_image, image_paths),
                             total=len(image_paths)):
                results.append(result)
        
        return results
    
    def generate_summary_stats(self, results):
        """Generate summary statistics from batch processing results"""
        summary = {
            'total_images': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'predictions': {
                'benign': sum(1 for r in results if 'error' not in r and r['prediction']['prediction'] == 0),
                'malignant': sum(1 for r in results if 'error' not in r and r['prediction']['prediction'] == 1)
            }
        }
        
        # Calculate average metrics
        if summary['successful'] > 0:
            valid_results = [r for r in results if 'error' not in r]
            
            # Color metrics
            color_means = {
                channel: np.mean([r['color_analysis'][channel]['mean'] for r in valid_results])
                for channel in ['R', 'G', 'B']
            }
            summary['average_color'] = color_means
            
            # Texture metrics
            texture_means = {
                feature: np.mean([r['texture_analysis'][feature] for r in valid_results])
                for feature in valid_results[0]['texture_analysis'].keys()
            }
            summary['average_texture'] = texture_means
        
        return summary
    
    def create_visualization_dashboard(self, results):
        """Create interactive visualizations for batch results"""
        valid_results = [r for r in results if 'error' not in r]
        
        # Prepare data for visualizations
        data = []
        for result in valid_results:
            row = {
                'filename': result['image_info']['filename'],
                'prediction': 'Malignant' if result['prediction']['prediction'] == 1 else 'Benign',
                'confidence': max(result['prediction']['probabilities']) * 100,
                'area': result['segmentation']['area'],
                'perimeter': result['segmentation']['perimeter']
            }
            # Add color metrics
            for channel, stats in result['color_analysis'].items():
                row[f'{channel}_mean'] = stats['mean']
                row[f'{channel}_std'] = stats['std']
            
            # Add texture metrics
            for feature, value in result['texture_analysis'].items():
                row[f'texture_{feature}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        visualizations = {}
        
        # Prediction distribution
        visualizations['prediction_dist'] = px.pie(
            df,
            names='prediction',
            title='Distribution of Predictions'
        )
        
        # Confidence histogram
        visualizations['confidence_hist'] = px.histogram(
            df,
            x='confidence',
            title='Distribution of Prediction Confidence'
        )
        
        # Feature correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        visualizations['correlation_matrix'] = px.imshow(
            correlation_matrix,
            title='Feature Correlation Matrix'
        )
        
        # Scatter plot matrix for key metrics
        key_metrics = ['area', 'perimeter', 'confidence']
        visualizations['scatter_matrix'] = px.scatter_matrix(
            df,
            dimensions=key_metrics,
            color='prediction',
            title='Scatter Plot Matrix of Key Metrics'
        )
        
        return visualizations
    
    def export_results(self, results, output_path):
        """Export batch analysis results to Excel file"""
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Summary sheet
            summary = self.generate_summary_stats(results)
            pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results
            valid_results = [r for r in results if 'error' not in r]
            
            # Predictions sheet
            predictions_data = [{
                'filename': r['image_info']['filename'],
                'prediction': 'Malignant' if r['prediction']['prediction'] == 1 else 'Benign',
                'confidence': max(r['prediction']['probabilities']) * 100
            } for r in valid_results]
            pd.DataFrame(predictions_data).to_excel(writer, sheet_name='Predictions', index=False)
            
            # Color analysis sheet
            color_data = []
            for r in valid_results:
                row = {'filename': r['image_info']['filename']}
                for channel, stats in r['color_analysis'].items():
                    for stat_name, value in stats.items():
                        row[f'{channel}_{stat_name}'] = value
                color_data.append(row)
            pd.DataFrame(color_data).to_excel(writer, sheet_name='Color_Analysis', index=False)
            
            # Texture analysis sheet
            texture_data = []
            for r in valid_results:
                row = {'filename': r['image_info']['filename']}
                row.update(r['texture_analysis'])
                texture_data.append(row)
            pd.DataFrame(texture_data).to_excel(writer, sheet_name='Texture_Analysis', index=False)
            
            # Errors sheet if any
            errors = [r for r in results if 'error' in r]
            if errors:
                pd.DataFrame(errors).to_excel(writer, sheet_name='Errors', index=False)
