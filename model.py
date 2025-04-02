import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import numpy as np

class ImageClassifier:
    def __init__(self):
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)
        self.model.eval()  # Set to evaluation mode
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load ImageNet class labels
        self.labels = self._load_imagenet_labels()
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels."""
        try:
            with open('imagenet_classes.txt') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print("Warning: imagenet_classes.txt not found. Using default labels.")
            return [f"class_{i}" for i in range(1000)]
    
    def preprocess_image(self, image):
        """Preprocess the input image for the model."""
        return self.transform(image)
    
    def process_uploaded_image(self, file_content: bytes) -> torch.Tensor:
        """
        Process an uploaded image file and convert it to a tensor.
        
        Args:
            file_content (bytes): The raw bytes of the uploaded image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
            
        Raises:
            ValueError: If the image cannot be processed
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing transformations
            tensor = self.preprocess_image(image)
            
            return tensor
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def process_image_array(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Process a numpy array image and convert it to a tensor.
        
        Args:
            image_array (np.ndarray): Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
            
        Raises:
            ValueError: If the image cannot be processed
        """
        try:
            # Convert numpy array to PIL Image
            if image_array.ndim == 2:  # Grayscale image
                image = Image.fromarray(image_array)
            else:  # RGB image
                image = Image.fromarray(image_array)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing transformations
            tensor = self.preprocess_image(image)
            
            return tensor
            
        except Exception as e:
            raise ValueError(f"Error processing image array: {str(e)}")
    
    def predict(self, image):
        """Make a prediction on the input image."""
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            return {
                "prediction": self.labels[predicted_idx],
                "confidence": confidence.item()
            }
    
    def predict_image(self, image_input):
        """
        Unified prediction function that accepts either a file upload or image array.
        
        Args:
            image_input: Either bytes (file upload) or numpy array
            
        Returns:
            dict: Prediction result containing prediction and confidence
            
        Raises:
            ValueError: If the input format is not supported or processing fails
        """
        try:
            # Process the image based on input type
            if isinstance(image_input, bytes):
                tensor = self.process_uploaded_image(image_input)
            elif isinstance(image_input, np.ndarray):
                tensor = self.process_image_array(image_input)
            else:
                raise ValueError("Unsupported input type. Must be bytes (file upload) or numpy array.")
            
            # Make prediction
            return self.predict(tensor)
            
        except Exception as e:
            raise ValueError(f"Error in predict_image: {str(e)}")

# Create a global instance of the classifier
classifier = ImageClassifier() 