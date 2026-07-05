import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import io

class ResnetEmbeddingExtractor:
    def __init__(self):
        # Load pre-trained ResNet18
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use updated torchvision weights API to avoid deprecation warnings
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer to get raw embeddings (512-dim)
        self.model.fc = nn.Identity()
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, image_path_or_bytes) -> list[float]:
        """
        Takes an image file path or raw bytes, and returns a 512-dim list of floats.
        """
        if isinstance(image_path_or_bytes, bytes):
            image = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
        else:
            image = Image.open(image_path_or_bytes).convert("RGB")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor)
            
        return embedding.squeeze(0).cpu().numpy().tolist()

# Singleton instance
_extractor = None

def get_extractor() -> ResnetEmbeddingExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ResnetEmbeddingExtractor()
    return _extractor
