import os
import requests
from pathlib import Path

def download_sample_images():
    """Download sample images for testing."""
    # Create sample_images directory
    image_dir = Path("sample_images")
    image_dir.mkdir(exist_ok=True)
    
    # Sample image URLs
    images = {
        "cat.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg",
        "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    }
    
    # Download each image
    for filename, url in images.items():
        print(f"Downloading {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(image_dir / filename, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

if __name__ == "__main__":
    print("Setting up test images...")
    download_sample_images()
    print("\nSetup complete! You can now run test_model.py") 