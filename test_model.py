import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from model import classifier

def test_with_local_image():
    """Test prediction with a local image file."""
    try:
        # Test with a local image file
        image_path = "sample_images/dog.jpg"  
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        result = classifier.predict_image(image_bytes)
        print("\nTest with local image:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        return True
    except Exception as e:
        print(f"Error testing with local image: {str(e)}")
        return False

def test_with_numpy_array():
    """Test prediction with a numpy array."""
    try:
        # Create a sample image array (224x224 RGB)
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result = classifier.predict_image(image_array)
        print("\nTest with numpy array:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        return True
    except Exception as e:
        print(f"Error testing with numpy array: {str(e)}")
        return False

def test_with_downloaded_image():
    """Test prediction with a downloaded image."""
    try:
        # Download a sample image from the internet
        url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        response = requests.get(url)
        image_bytes = response.content
        
        result = classifier.predict_image(image_bytes)
        print("\nTest with downloaded image:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        return True
    except Exception as e:
        print(f"Error testing with downloaded image: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Starting model tests...")
    
    # Create sample_images directory if it doesn't exist
    os.makedirs("sample_images", exist_ok=True)
    
    # Run tests
    local_test = test_with_local_image()
    numpy_test = test_with_numpy_array()
    download_test = test_with_downloaded_image()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Local image test: {'✓' if local_test else '✗'}")
    print(f"Numpy array test: {'✓' if numpy_test else '✗'}")
    print(f"Downloaded image test: {'✓' if download_test else '✗'}")

if __name__ == "__main__":
    main() 