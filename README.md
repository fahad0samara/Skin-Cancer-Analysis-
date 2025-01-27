# Skin Cancer Analysis System

A comprehensive system for analyzing skin cancer images using deep learning and computer vision techniques. Built with clean architecture principles and modern AI technologies.

## Features

- **Image Analysis**: Advanced skin cancer image analysis using deep learning
- **Multi-class Classification**: Support for multiple skin cancer types
- **Model Training**: Custom model training with various architectures
- **REST API**: FastAPI-based REST API for easy integration
- **Clean Architecture**: Well-organized codebase following clean architecture principles
- **Data Management**: Efficient data storage and retrieval system
- **Performance Metrics**: Comprehensive model evaluation and metrics tracking

## Project Structure

```
src/
├── domain/              # Enterprise Business Rules
│   ├── interfaces.py    # Abstract interfaces
│   └── use_cases.py     # Business logic use cases
├── infrastructure/      # Frameworks & Drivers
│   └── repositories.py  # Data storage implementations
├── application/        # Application Business Rules
│   └── services.py     # Application services
└── presentation/       # Interface Adapters
    └── api.py         # REST API endpoints
```

## Installation

1. Clone the repository:
```bash
cd skin-cancer-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

```bash
python -m src.presentation.api
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /upload`: Upload a new image for analysis
- `POST /predict/{image_id}`: Get prediction for a specific image
- `POST /train`: Train a new model
- `GET /analyze`: Analyze the current dataset
- `GET /models/{model_id}/metrics`: Get model metrics
- `GET /predictions`: Get prediction history

### Example Usage

```python
import requests

# Upload an image
with open('skin_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
image_id = response.json()['image_id']

# Get prediction
prediction = requests.post(
    f'http://localhost:8000/predict/{image_id}',
    json={'model_id': 'latest'}
).json()

print(f"Prediction: {prediction['predicted_class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## Model Training

To train a new model:

```python
from src.application.services import ModelTrainingService
from src.infrastructure.repositories import (
    FileSystemImageRepository,
    FileSystemModelRepository
)

# Initialize repositories
image_repo = FileSystemImageRepository("data/images")
model_repo = FileSystemModelRepository("data/models")

# Initialize training service
training_service = ModelTrainingService(image_repo, model_repo)

# Train model
metrics = training_service.train_model(
    model_architecture="efficientnet",
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001
)

print(f"Training accuracy: {metrics.accuracy:.2%}")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guide. To check code style:

```bash
flake8 src/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ISIC Archive for providing the skin cancer image dataset
- EfficientNet team for the model architecture
- FastAPI team for the excellent web framework
