import torch
import torchvision.models as models
from torchvision import transforms

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

# Create a global instance of the classifier
classifier = ImageClassifier() 