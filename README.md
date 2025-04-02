# Image Classification API

A collaborative FastAPI project that provides image classification using a pre-trained ResNet18 model.

## Project Structure

The project is organized into two main branches:

### Branch-a (Mahmoud) (API Endpoints)
- `main.py` - FastAPI application with endpoints
  - `/status` - GET endpoint to check API health
  - `/predict` - POST endpoint for image classification

### Branch-b (Hussein) (Model Implementation)
- `model.py` - Image classification implementation
  - Pre-trained ResNet18 model
  - Image preprocessing utilities
  - Prediction functionality
- `test_model.py` - Test script for model validation
- `setup_test_images.py` - Script to download test images

### Web UI
- `app.py` - Flask web application for image upload and classification
- `templates/index.html` - Web interface template for uploading images

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
- `flask>=3.0.0` - Flask web framework
- `pytest>=8.0.0` - Testing framework
- `httpx>=0.20.0` - HTTP client for testing
- `werkzeug>=3.0.0` - WSGI utilities

## Usage

### Running the Backend API

1. Start the FastAPI server:
```bash
python -m uvicorn main:app --reload
```
The server will start at `http://localhost:8000`

### Running the Web Interface

1. Start the Flask web application (in a separate terminal):
```bash
python app.py
```
The web interface will be available at `http://localhost:5000`

2. Access the web interface in your browser, upload an image, and get AI predictions

### Testing the Model Directly

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
- API endpoint testing
- Model prediction testing
- Web interface testing

Run API tests with:
```bash
pytest test_main.py
```

Run model tests with:
```bash
python test_model.py
```

Run all tests:
```bash
pytest
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
  
