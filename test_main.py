from fastapi.testclient import TestClient
import io
from main import app

client = TestClient(app)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
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