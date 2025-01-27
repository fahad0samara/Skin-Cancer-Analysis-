import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms, models
import cv2
import matplotlib.pyplot as plt
import io

class SkinCancerDetection:
    def __init__(self):
        """Initialize the skin cancer detection application."""
        st.set_page_config(
            page_title="Skin Cancer Detection",
            page_icon="🔬",
            layout="wide"
        )
        
        self.setup_model()
        self.setup_transforms()
    
    def setup_model(self):
        """Setup the model for skin cancer classification."""
        # Use ResNet50 as the base model
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # Binary classification: malignant vs benign
        
        # Set model to evaluation mode
        self.model.eval()
        
        self.classes = [
            'Benign',
            'Malignant (Skin Cancer)'
        ]
        
        # ABCD criteria thresholds
        self.abcd_thresholds = {
            'asymmetry': 0.7,
            'border': 0.7,
            'color': 0.7,
            'diameter': 6.0  # in mm
        }
    
    def setup_transforms(self):
        """Setup image transformations."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def analyze_abcd(self, image, mask):
        """Analyze ABCD criteria of the lesion."""
        results = {}
        
        # Analyze Asymmetry
        results['asymmetry'] = self.analyze_asymmetry(mask)
        
        # Analyze Border
        results['border'] = self.analyze_border(mask)
        
        # Analyze Color
        results['color'] = self.analyze_color(image, mask)
        
        # Analyze Diameter
        results['diameter'] = self.analyze_diameter(mask)
        
        return results
    
    def analyze_asymmetry(self, mask):
        """Analyze asymmetry of the lesion."""
        # Split mask into left and right halves
        h, w = mask.shape
        left = mask[:, :w//2]
        right = np.fliplr(mask[:, w//2:])
        
        # Calculate asymmetry score
        asymmetry = np.sum(np.abs(left - right)) / (h * w/2)
        return float(asymmetry)
    
    def analyze_border(self, mask):
        """Analyze border irregularity."""
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            # Calculate perimeter and area
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            
            # Calculate border irregularity (circularity)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return 1 - circularity  # Higher value means more irregular
        return 0.0
    
    def analyze_color(self, image, mask):
        """Analyze color variation in the lesion."""
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate color variation
        valid_pixels = masked[mask > 0]
        if len(valid_pixels) > 0:
            std_devs = np.std(valid_pixels, axis=0)
            return float(np.mean(std_devs) / 255.0)
        return 0.0
    
    def analyze_diameter(self, mask):
        """Analyze diameter of the lesion."""
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            # Find minimum enclosing circle
            _, radius = cv2.minEnclosingCircle(contours[0])
            # Convert radius to mm (approximate)
            diameter_mm = 2 * radius * 0.1  # assuming 1 pixel ≈ 0.1mm
            return float(diameter_mm)
        return 0.0
    
    def segment_lesion(self, image):
        """Segment the skin lesion from the image."""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def create_visualization(self, image, mask, abcd_results):
        """Create visualization of the analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Segmentation mask
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Lesion Segmentation')
        ax2.axis('off')
        
        # ABCD Criteria Plot
        criteria = ['Asymmetry', 'Border', 'Color', 'Diameter']
        values = [
            abcd_results['asymmetry'],
            abcd_results['border'],
            abcd_results['color'],
            min(1.0, abcd_results['diameter'] / 6.0)  # Normalize diameter
        ]
        
        bars = ax3.bar(criteria, values)
        ax3.set_title('ABCD Criteria Analysis')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Risk Assessment
        risk_score = np.mean(values) * 100
        ax4.text(0.5, 0.5,
                f'Risk Assessment:\n\n'
                f'Overall Risk Score: {risk_score:.1f}%\n\n'
                f'{"❗ High Risk" if risk_score > 60 else "✓ Low Risk"}',
                ha='center', va='center',
                fontsize=12)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def generate_recommendations(self, abcd_results, risk_score):
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # High risk indicators
        high_risk = False
        risk_factors = []
        
        # Check each ABCD criterion
        if abcd_results['asymmetry'] > self.abcd_thresholds['asymmetry']:
            risk_factors.append("high asymmetry")
        
        if abcd_results['border'] > self.abcd_thresholds['border']:
            risk_factors.append("irregular borders")
        
        if abcd_results['color'] > self.abcd_thresholds['color']:
            risk_factors.append("concerning color variations")
        
        if abcd_results['diameter'] > self.abcd_thresholds['diameter']:
            risk_factors.append(f"large diameter ({abcd_results['diameter']:.1f}mm)")
        
        # Generate recommendations based on risk factors
        if risk_factors:
            high_risk = True
            risk_str = ", ".join(risk_factors[:-1])
            if len(risk_factors) > 1:
                risk_str += f", and {risk_factors[-1]}"
            else:
                risk_str = risk_factors[0]
                
            recommendations.append(
                f"⚠️ This lesion shows {risk_str}, which may indicate increased "
                "risk of skin cancer."
            )
            recommendations.append(
                "👨‍⚕️ Immediate consultation with a dermatologist is strongly recommended."
            )
        else:
            recommendations.append(
                "✅ No high-risk features were detected in this lesion."
            )
            recommendations.append(
                "👍 While the analysis suggests low risk, continue regular skin "
                "self-examinations and annual check-ups with a healthcare provider."
            )
        
        # Additional recommendations
        recommendations.append(
            "📝 Keep track of any changes in size, shape, color, or texture of "
            "this and other skin lesions."
        )
        recommendations.append(
            "🔍 Regular skin self-examinations are recommended to detect any "
            "changes early."
        )
        
        return recommendations, high_risk
    
    def run(self):
        """Run the application."""
        st.title("🔬 Skin Cancer Detection")
        
        st.markdown("""
            This application analyzes skin lesions for potential signs of skin cancer
            using the ABCD criteria:
            - **A**symmetry
            - **B**order irregularity
            - **C**olor variation
            - **D**iameter
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload an image of the skin lesion",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            
            with st.spinner("Analyzing image..."):
                # Perform segmentation
                mask = self.segment_lesion(image)
                
                # Analyze ABCD criteria
                abcd_results = self.analyze_abcd(image, mask)
                
                # Calculate overall risk score
                risk_score = np.mean([
                    abcd_results['asymmetry'],
                    abcd_results['border'],
                    abcd_results['color'],
                    min(1.0, abcd_results['diameter'] / 6.0)
                ]) * 100
                
                # Create and display visualization
                viz = self.create_visualization(image, mask, abcd_results)
                st.image(viz, use_column_width=True)
                
                # Generate and display recommendations
                recommendations, high_risk = self.generate_recommendations(
                    abcd_results, risk_score
                )
                
                # Display recommendations
                st.subheader("Analysis Results & Recommendations")
                
                # Risk level indicator
                if high_risk:
                    st.error("🚨 High Risk Detected")
                else:
                    st.success("✅ Low Risk Assessment")
                
                # Display detailed recommendations
                for rec in recommendations:
                    st.write(rec)
                
                # Disclaimer
                st.warning(
                    "⚠️ **Disclaimer**: This tool is for educational purposes only "
                    "and should not replace professional medical advice. Always "
                    "consult with a qualified healthcare provider for proper "
                    "diagnosis and treatment."
                )

if __name__ == "__main__":
    app = SkinCancerDetection()
    app.run()
