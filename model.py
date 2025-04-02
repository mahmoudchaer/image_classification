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
        # Common ImageNet classes
        common_classes = [
            'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 
            'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 
            'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 
            'bald eagle', 'vulture', 'great grey owl', 'European fire salamander', 'common newt', 'eft', 
            'spotted salamander', 'axolotl', 'bullfrog', 'tree frog', 'tailed frog', 'loggerhead', 
            'leatherback turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana', 
            'American chameleon', 'whiptail', 'agama', 'frilled lizard', 'alligator lizard', 'Gila monster', 
            'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile', 'American alligator', 
            'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake', 'king snake', 
            'garter snake', 'water snake', 'vine snake', 'night snake', 'boa constrictor', 'rock python', 
            'Indian cobra', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder', 'trilobite', 
            'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 
            'black widow', 'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 
            'ruffed grouse', 'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey', 'macaw', 
            'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 
            'toucan', 'drake', 'red-breasted merganser', 'goose', 'black swan', 'tusker', 'echidna', 'platypus', 
            'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 
            'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab', 
            'fiddler crab', 'king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 
            'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'American egret', 
            'bittern', 'crane', 'limpkin', 'European gallinule', 'American coot', 'bustard', 'ruddy turnstone', 
            'red-backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 
            'grey whale', 'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog', 
            'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 
            'Afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan coonhound', 
            'Walker hound', 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 
            'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 
            'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 
            'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 
            'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 
            'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 
            'standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier', 
            'West Highland white terrier', 'Lhasa', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 
            'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla', 'English setter', 
            'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer', 'Welsh springer spaniel', 
            'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 
            'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie', 
            'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher', 
            'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 
            'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky', 
            'dalmatian', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 
            'Pomeranian', 'chow', 'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 
            'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 'red wolf', 
            'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox', 
            'grey fox', 'tabby', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'cougar', 'lynx', 
            'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 
            'ice bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle', 
            'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 
            'cricket', 'walking stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 
            'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid', 
            'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit', 'hare', 'Angora', 'hamster', 'porcupine', 
            'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'sorrel', 'zebra', 'hog', 'wild boar', 'warthog', 
            'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle'
        ]
        
        # If the list doesn't contain 1000 elements, pad with class_X
        if len(common_classes) < 1000:
            for i in range(len(common_classes), 1000):
                common_classes.append(f"class_{i}")
                
        return common_classes
    
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