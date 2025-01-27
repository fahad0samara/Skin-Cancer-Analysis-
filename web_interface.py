from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from skin_cancer_model import SkinCancerModel
import base64
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model
model = SkinCancerModel()
model_path = 'model.pth'
if os.path.exists(model_path):
    model.load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            results = model.predict_image(filepath)
            
            # Get explanation
            explanations = model.get_prediction_explanation(results['abcd_results'])
            
            # Convert image to base64 for display
            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': 'Malignant' if results['prediction'] == 1 else 'Benign',
                'confidence': float(max(results['probabilities']) * 100),
                'explanations': explanations,
                'abcd_results': results['abcd_results'],
                'image': encoded_image
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Failed to process file'})

if __name__ == '__main__':
    app.run(debug=True)
