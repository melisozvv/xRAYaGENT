"""
TorchXrayVision Classification Tool - Pathology Detection
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class TorchXrayVisionClassifier:
    """TorchXrayVision tool for chest X-ray pathology classification"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.pathologies = []
        
    def load_model(self, model_type: str = "densenet121-res224-all"):
        """Load the TorchXrayVision model"""
        try:
            import torchxrayvision as xrv
            import torch
            import torchvision
            
            # Load model
            self.model = xrv.models.DenseNet(weights=model_type)
            self.model.eval()
            
            # Setup transforms
            self.transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])
            
            # Get pathologies for this model
            self.pathologies = self.model.pathologies
            
            logger.info(f"TorchXrayVision model loaded: {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TorchXrayVision model: {e}")
            return False
    
    def classify_pathologies(self, image_path: str, model_type: str = "densenet121-res224-all", 
                           threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classify pathologies in chest X-ray
        
        Args:
            image_path: Path to the X-ray image
            model_type: Type of model to use
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with pathology predictions
        """
        try:
            import torchxrayvision as xrv
            import torch
            import skimage
            
            # Load and preprocess image
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            img = skimage.io.imread(image_path)
            img = xrv.datasets.normalize(img, 255)
            
            # Handle different image formats
            if len(img.shape) == 3:
                img = img.mean(2)  # Convert to grayscale
            img = img[None, ...]
            
            # Ensure model is loaded
            if not self.model or self.model.weights != model_type:
                if not self.load_model(model_type):
                    return {"error": "Failed to load TorchXrayVision model"}
            
            # Apply transforms
            img = self.transform(img)
            img = torch.from_numpy(img).float()
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(img[None, ...])
                predictions = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Create results dictionary
            pathology_scores = dict(zip(self.pathologies, predictions))

            return {
                "pathology_scores": pathology_scores,
                "image_path": image_path,
                "tool_name": "TorchXrayVision"
            }
            
        except Exception as e:
            logger.error(f"Error in TorchXrayVision classification: {e}")
            return {"error": str(e)}
    
    def batch_classify(self, image_paths: List[str], model_type: str = "densenet121-res224-all") -> Dict[str, Any]:
        """
        Classify multiple images
        
        Args:
            image_paths: List of paths to X-ray images
            model_type: Type of model to use
            
        Returns:
            Dictionary with batch results
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            result = self.classify_pathologies(image_path, model_type)
            results[f"image_{i+1}"] = result
        
        return {
            "batch_results": results,
            "total_images": len(image_paths),
            "model_type": model_type,
            "tool_name": "TorchXrayVision"
        }
    
    def compare_models(self, image_path: str, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Compare different model predictions on the same image
        
        Args:
            image_path: Path to the X-ray image
            model_types: List of model types to compare
            
        Returns:
            Dictionary with comparison results
        """
        if not model_types:
            model_types = [
                "densenet121-res224-all",
                "densenet121-res224-rsna", 
                "densenet121-res224-nih",
                "densenet121-res224-chex"
            ]
        
        comparison_results = {}
        
        for model_type in model_types:
            result = self.classify_pathologies(image_path, model_type)
            comparison_results[model_type] = result
        
        return {
            "model_comparison": comparison_results,
            "image_path": image_path,
            "models_compared": model_types,
            "tool_name": "TorchXrayVision"
        }
    
    def get_model_info(self, model_type: str = "densenet121-res224-all") -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_type: Type of model to get info for
            
        Returns:
            Dictionary with model information
        """
        model_info = {
            "densenet121-res224-all": {
                "description": "DenseNet121 trained on multiple datasets",
                "datasets": ["NIH", "CheXpert", "MIMIC-CXR", "PadChest"],
                "pathologies": [
                    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
                    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
                    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
                ]
            },
            "densenet121-res224-rsna": {
                "description": "DenseNet121 trained on RSNA pneumonia dataset",
                "datasets": ["RSNA"],
                "pathologies": ["Lung Opacity"]
            },
            "densenet121-res224-nih": {
                "description": "DenseNet121 trained on NIH dataset",
                "datasets": ["NIH"],
                "pathologies": [
                    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
                    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
                ]
            },
            "densenet121-res224-chex": {
                "description": "DenseNet121 trained on CheXpert dataset",
                "datasets": ["CheXpert"],
                "pathologies": [
                    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                    "Effusion", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
                    "Lung Opacity", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
                ]
            }
        }
        
        return model_info.get(model_type, {"description": "Unknown model", "datasets": [], "pathologies": []})

# Standalone testing functions
def test_torchxrayvision():
    """Test function for TorchXrayVision"""
    print("Testing TorchXrayVision...")
    
    # Find test image
    test_image_paths = [
        "../../data/xray.jpg",
    ]
    
    image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print("❌ No test image found")
        return False
    
    # Initialize tool
    classifier = TorchXrayVisionClassifier()
    
    # Test classification
    result = classifier.classify_pathologies(image_path)
    
    print("✅ TorchXrayVision test passed")
    print(f"Top predictions: {result['pathology_scores']}")
    return True

if __name__ == "__main__":
    test_torchxrayvision() 