from fastapi.testclient import TestClient
import io
import unittest.mock as mock
from main import app

client = TestClient(app)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@mock.patch('main.classifier.predict_image')
def test_predict(mock_predict):
    # Mock the classifier's predict_image method
    mock_predict.return_value = {"prediction": "dog", "confidence": 0.99}
    
    # Create a test file
    test_file = io.BytesIO(b"test image content")
    test_file.name = "test_image.jpg"
    
    # Test the predict endpoint
    response = client.post(
        "/predict",
        files={"file": (test_file.name, test_file, "image/jpeg")}
    )
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
    assert response.json()["prediction"] == "dog"
    assert response.json()["confidence"] == 0.99
    
    # Verify the mock was called
    mock_predict.assert_called_once() 