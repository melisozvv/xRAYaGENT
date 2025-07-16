import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class COVID19Detection:
    """Tool for detecting COVID-19 in chest X-ray images using BEiT model"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False
        self.model_name = "Jesteban247/beit-base-patch16-224-pt22k-ft22k-finetuned-Chest_Xray_COVID19"
        self.class_labels = None
        
    def load_model(self, model_name: str = None):
        """Load the COVID-19 detection BEiT model from Hugging Face"""
        try:
            from transformers import BeitImageProcessor, BeitForImageClassification
            import torch
            
            if model_name:
                self.model_name = model_name
            
            logger.info(f"Loading COVID-19 detection model: {self.model_name}")
            
            # Load processor and model
            self.processor = BeitImageProcessor.from_pretrained(self.model_name)
            self.model = BeitForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # Store actual class labels from model config
            self.class_labels = self.model.config.id2label
            
            self.loaded = True
            logger.info(f"COVID-19 detection model loaded successfully: {self.model_name}")
            logger.info(f"Available classes: {list(self.class_labels.values())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load COVID-19 detection model: {e}")
            self.loaded = False
            return False
    
    def detect_covid19(self, image_path: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Detect COVID-19 in chest X-ray image
        
        Args:
            image_path: Path to the X-ray image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with COVID-19 detection results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        # Ensure model is loaded
        if not self.loaded:
            if not self.load_model():
                return {"error": "Failed to load COVID-19 detection model"}
        
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Get predictions
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = logits.argmax().item()
            
            # Get class labels
            class_labels = self.model.config.id2label
            predicted_label = class_labels[predicted_class_id]
            
            # Get probability for each class
            class_probabilities = {}
            for class_id, label in class_labels.items():
                class_probabilities[label] = float(probabilities[0][class_id])
        
        result = {
            "predicted_class": predicted_label,
            "predicted_class_id": predicted_class_id,
            "confidence": float(probabilities[0][predicted_class_id]),
            "image_path": image_path,
            "tool_name": "COVID19Detection",
            "model_name": self.model_name
        }
        
        if return_probabilities:
            result["class_probabilities"] = class_probabilities
        
        return result
        
    
    def batch_detect(self, image_paths: List[str], return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Detect COVID-19 in multiple images
        
        Args:
            image_paths: List of paths to X-ray images
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with batch results
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            result = self.detect_covid19(image_path, return_probabilities)
            results[f"image_{i+1}"] = result
        
        return {
            "batch_results": results,
            "total_images": len(image_paths),
            "model_name": self.model_name,
            "tool_name": "COVID19Detection"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the COVID-19 detection model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "base_model": "microsoft/beit-base-patch16-224-pt22k-ft22k",
            "description": "BEiT model fine-tuned for COVID-19 detection in chest X-rays",
            "input_size": "224x224",
            "model_type": "Vision Transformer (BEiT)",
            "classes": list(self.class_labels.values()) if self.loaded and self.class_labels else "Model not loaded",
            "architecture": "BEiT (Bidirectional Encoder representation from Image Transformers)",
            "fine_tuned_on": "Chest X-ray COVID-19 dataset",
            "preprocessing": "BeitImageProcessor with normalization and resizing to 224x224"
        }
    
    def analyze_covid_risk(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze COVID-19 risk with detailed interpretation
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            Dictionary with detailed COVID-19 risk analysis
        """
        # Get base detection results
        result = self.detect_covid19(image_path, return_probabilities=True)
        
        if "error" in result:
            return result
        
        # Extract probabilities
        probs = result["class_probabilities"]
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        
        # Find COVID-related class dynamically
        covid_prob = 0.0
        covid_class_name = None
        
        # Look for COVID-related classes (case-insensitive)
        for class_name, prob in probs.items():
            if "covid" in class_name.lower():
                covid_prob = prob
                covid_class_name = class_name
                break
        
        if covid_prob >= 0.8:
            risk_level = "High"
            interpretation = "Strong indication of COVID-19 patterns"
        elif covid_prob >= 0.6:
            risk_level = "Moderate-High"
            interpretation = "Moderate to high likelihood of COVID-19"
        elif covid_prob >= 0.4:
            risk_level = "Moderate"
            interpretation = "Moderate probability of COVID-19"
        elif covid_prob >= 0.2:
            risk_level = "Low-Moderate"
            interpretation = "Low to moderate probability of COVID-19"
        else:
            risk_level = "Low"
            interpretation = "Low probability of COVID-19"
        
        return {
            "covid_probability": covid_prob,
            "covid_class_name": covid_class_name,
            "risk_level": risk_level,
            "interpretation": interpretation,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": probs,
            "image_path": image_path,
            "tool_name": "COVID19Detection",
            "model_name": self.model_name,
            "recommendation": "Consult with healthcare professional for clinical interpretation",
            "note": "No COVID-related class found in model" if covid_class_name is None else None
        }
            


if __name__ == "__main__":
    # COVID-19 Detection using BEiT Model
    # Model: https://huggingface.co/Jesteban247/beit-base-patch16-224-pt22k-ft22k-finetuned-Chest_Xray_COVID19
    # Base Model: https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k
    covid_detector = COVID19Detection()
    test_image_path = "../../data/xray.jpg"

    # Perform COVID-19 detection
    if os.path.exists(test_image_path):
        print("Performing COVID-19 detection...")
        
        # Basic detection
        result = covid_detector.detect_covid19(test_image_path)
        
        if "error" not in result:
            print("\n=== COVID-19 Detection Results ===")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Model Used: {result['model_name']}")