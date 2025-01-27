from skin_cancer_model import SkinCancerModel, SkinCancerDataset
from analysis.multiclass_skin_cancer_analysis import MulticlassSkinCancerAnalysis
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Detection Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict'],
                      help='Mode to run the model in')
    parser.add_argument('--model_path', type=str, default='model.pth',
                      help='Path to save/load the model')
    parser.add_argument('--image_path', type=str,
                      help='Path to image for prediction')
    args = parser.parse_args()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = SkinCancerModel()
    
    if args.mode == 'train':
        # Set data directories
        train_dir = "Train"
        test_dir = "Test"
        
        # Check if directories exist
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print(f"Error: Make sure both {train_dir} and {test_dir} directories exist!")
            return
        
        # Load and train model
        print("\nLoading training data...")
        train_paths, train_labels = model.load_data(train_dir)
        print(f"Found {len(train_paths)} training images")
        
        # Print class distribution
        num_malignant = sum(train_labels)
        num_benign = len(train_labels) - num_malignant
        print(f"Class distribution:")
        print(f"- Benign: {num_benign} images")
        print(f"- Malignant: {num_malignant} images")
        
        # Create dataset and train
        print("\nPreparing data loaders...")
        train_dataset = SkinCancerDataset(train_paths, train_labels, transform=model.transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # Train the model
        print("\nStarting training...")
        history = model.train_model(train_loader, num_epochs=10)
        
        # Save the model
        model.save_model(args.model_path)
        
        # Plot results
        print("\nPlotting training results...")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'])
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.show()
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_paths, test_labels = model.load_data(test_dir)
        test_dataset = SkinCancerDataset(test_paths, test_labels, transform=model.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        model.evaluate_model(test_loader)
        
    elif args.mode == 'test':
        # Load the model
        if not model.load_model(args.model_path):
            return
            
        # Load test data
        test_dir = "Test"
        test_paths, test_labels = model.load_data(test_dir)
        test_dataset = SkinCancerDataset(test_paths, test_labels, transform=model.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Evaluate
        model.evaluate_model(test_loader)
        
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: Please provide an image path using --image_path")
            return
            
        if not os.path.exists(args.image_path):
            print(f"Error: Image not found at {args.image_path}")
            return
            
        # Load the model
        if not model.load_model(args.model_path):
            return
            
        # Make prediction
        results = model.predict_image(args.image_path)
        
        prediction = "Malignant" if results['prediction'] == 1 else "Benign"
        confidence = max(results['probabilities']) * 100
        
        print(f"\nPrediction: {prediction} (Confidence: {confidence:.1f}%)")
        
        # Get and print explanation
        print("\nAnalysis Explanation:")
        explanations = model.get_prediction_explanation(results['abcd_results'])
        for explanation in explanations:
            print(f"- {explanation}")
        
        # Visualize results
        model.visualize_prediction(args.image_path, results)

    elif args.mode == 'multiclass':
        # Set paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_data_path = os.path.join(current_dir, 'train')
        model_path = os.path.join(current_dir, 'models', 'skin_cancer_model.h5')
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(current_dir, 'models'), exist_ok=True)
        
        # Initialize analyzer
        print("Initializing Skin Cancer Analysis...")
        analyzer = MulticlassSkinCancerAnalysis(train_data_path)
        
        # Run complete analysis
        print("\nRunning complete analysis...")
        model = analyzer.run_complete_analysis(epochs=20, batch_size=32)
        
        # Save the model
        print("\nSaving model...")
        analyzer.save_model(model_path)
        
        print("\nAnalysis complete! Model saved to:", model_path)

if __name__ == "__main__":
    main()
