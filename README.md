# Image Classification API

A collaborative FastAPI project that provides image classification using a pre-trained ResNet18 model.

## Project Structure

The project is organized into two main branches:

### Branch-a (API Endpoints)
- `main.py` - FastAPI application with endpoints
  - `/status` - GET endpoint to check API health
  - `/predict` - POST endpoint for image classification

### Branch-b (Model Implementation)
- `model.py` - Image classification implementation
  - Pre-trained ResNet18 model
  - Image preprocessing utilities
  - Prediction functionality
- `test_model.py` - Test script for model validation
- `setup_test_images.py` - Script to download test images

## Setup & Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download test images:
```bash
python setup_test_images.py
```

## Dependencies

- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - Computer vision utilities
- `fastapi>=0.68.0` - FastAPI framework
- `uvicorn>=0.15.0` - ASGI server
- `python-multipart>=0.0.5` - Multipart form data parsing
- `pillow>=8.0.0` - Image processing
- `requests>=2.31.0` - HTTP requests for testing

## Usage

1. Start the API server:
```bash
python main.py
```
The server will start at `http://localhost:8000`

2. Test the model:
```bash
python test_model.py
```

## API Endpoints

### GET /status
Check if the API is running:
```bash
curl http://localhost:8000/status
```
Response:
```json
{"status": "ok"}
```

### POST /predict
Upload an image for classification:
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/predict
```
Response:
```json
{
    "prediction": "golden retriever",
    "confidence": 0.93
}
```

## Testing

The test suite includes:
- Local image testing
- Numpy array testing
- Downloaded image testing

Run tests with:
```bash
python test_model.py
```

## Model Details

- Architecture: ResNet18
- Pre-trained: Yes (ImageNet)
- Input size: 224x224 pixels
- Output: 1000 ImageNet classes
- Image preprocessing:
  - Resize to 256px
  - Center crop to 224px
  - Normalize with ImageNet stats

## Contributors

- User A: API Implementation (branch-a)
- User B: Model Implementation (branch-b)

## License

[Add your license information here]
  