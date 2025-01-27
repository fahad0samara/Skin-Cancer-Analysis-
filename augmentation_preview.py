import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from flask import Blueprint, request, jsonify

augmentation_bp = Blueprint('augmentation', __name__)

class AugmentationPreview:
    def __init__(self):
        self.augmentations = {
            'original': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]),
            'flip': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]),
            'rotate': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]),
            'color': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ])
        }

    def apply_augmentation(self, image, aug_type):
        if aug_type not in self.augmentations:
            return None
        
        transform = self.augmentations[aug_type]
        augmented = transform(image)
        
        # Convert to base64
        buffered = io.BytesIO()
        augmented.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str

preview = AugmentationPreview()

@augmentation_bp.route('/preview_augmentation', methods=['POST'])
def preview_augmentation():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    aug_type = request.form.get('type', 'original')
    
    if file:
        try:
            # Open and process image
            image = Image.open(file).convert('RGB')
            augmented_b64 = preview.apply_augmentation(image, aug_type)
            
            if augmented_b64 is None:
                return jsonify({'error': 'Invalid augmentation type'})
            
            return jsonify({
                'success': True,
                'image': augmented_b64
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Failed to process file'})
