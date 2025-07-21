import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class COVID19GoogleDetection:
    """Tool for detecting COVID-19 in chest X-ray images using Google's model from yuighj123/image_classification_covid19"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False
        self.model_name = "yuighj123/image_classification_covid19"
        self.class_labels = None
        
    def load_model(self, model_name: str = None):
        """Load the COVID-19 detection model from Hugging Face"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            
            if model_name:
                self.model_name = model_name
            
            logger.info(f"Loading COVID-19 Google detection model: {self.model_name}")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # Store actual class labels from model config
            self.class_labels = self.model.config.id2label
            
            self.loaded = True
            logger.info(f"COVID-19 Google detection model loaded successfully: {self.model_name}")
            logger.info(f"Available classes: {list(self.class_labels.values())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load COVID-19 Google detection model: {e}")
            self.loaded = False
            return False
    
    def detect_covid19(self, image_path: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Detect COVID-19 in chest X-ray image using Google's model
        
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
                return {"error": "Failed to load COVID-19 Google detection model"}
        
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
            print(logits)
            
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
            "tool_name": "COVID19GoogleDetection",
            "model_name": self.model_name
        }
        
        if return_probabilities:
            result["class_probabilities"] = class_probabilities
        
        return result
        
    
    def batch_detect(self, image_paths: List[str], return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Detect COVID-19 in multiple images using Google's model
        
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
            "tool_name": "COVID19GoogleDetection"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the COVID-19 Google detection model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "base_model": "Google's image classification model",
            "description": "Google's model fine-tuned for COVID-19 detection in chest X-rays",
            "input_size": "224x224",
            "model_type": "Vision Transformer (Google)",
            "classes": list(self.class_labels.values()) if self.loaded and self.class_labels else "Model not loaded",
            "architecture": "Google's Vision Transformer architecture",
            "fine_tuned_on": "Chest X-ray COVID-19 dataset",
            "preprocessing": "AutoImageProcessor with normalization and resizing to 224x224",
            "source": "https://huggingface.co/yuighj123/image_classification_covid19"
        }
    
    def analyze_covid_risk(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze COVID-19 risk with detailed interpretation using Google's model
        
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
        
        # If no COVID class found, use the predicted class probability
        if covid_class_name is None:
            covid_prob = confidence
            covid_class_name = predicted_class
        
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
            "tool_name": "COVID19GoogleDetection",
            "model_name": self.model_name,
            "recommendation": "Consult with healthcare professional for clinical interpretation",
            "note": "No COVID-related class found in model" if covid_class_name is None else None
        }
    
    def compare_with_original_model(self, image_path: str) -> Dict[str, Any]:
        """
        Compare results with the original COVID19Detection model
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Import the original model for comparison
            from .covid19 import COVID19Detection
            
            # Get results from both models
            google_result = self.detect_covid19(image_path, return_probabilities=True)
            original_detector = COVID19Detection()
            original_result = original_detector.detect_covid19(image_path, return_probabilities=True)
            
            comparison = {
                "google_model": {
                    "predicted_class": google_result.get("predicted_class"),
                    "confidence": google_result.get("confidence"),
                    "model_name": google_result.get("model_name")
                },
                "original_model": {
                    "predicted_class": original_result.get("predicted_class"),
                    "confidence": original_result.get("confidence"),
                    "model_name": original_result.get("model_name")
                },
                "agreement": google_result.get("predicted_class") == original_result.get("predicted_class"),
                "confidence_difference": abs(google_result.get("confidence", 0) - original_result.get("confidence", 0)),
                "image_path": image_path
            }
            
            return comparison
            
        except Exception as e:
            return {
                "error": f"Failed to compare models: {e}",
                "google_result": self.detect_covid19(image_path, return_probabilities=True)
            }
            


if __name__ == "__main__":
    # COVID-19 Detection using Google's Model
    # Model: https://huggingface.co/yuighj123/image_classification_covid19
    covid_google_detector = COVID19GoogleDetection()
    test_image_path = "../../data/xray.jpg"

    # Perform COVID-19 detection
    if os.path.exists(test_image_path):
        print("Performing COVID-19 detection with Google's model...")
        
        # Basic detection
        result = covid_google_detector.detect_covid19(test_image_path)
        print(result)
        
        if "error" not in result:
            print("\n=== COVID-19 Google Detection Results ===")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Model Used: {result['model_name']}")
            
            # Risk analysis
            risk_result = covid_google_detector.analyze_covid_risk(test_image_path)
            print(f"\n=== Risk Analysis ===")
            print(f"Risk Level: {risk_result['risk_level']}")
            print(f"COVID Probability: {risk_result['covid_probability']:.4f}")
            print(f"Interpretation: {risk_result['interpretation']}")
            
            # Model comparison (if original model is available)
            try:
                comparison = covid_google_detector.compare_with_original_model(test_image_path)
                if "error" not in comparison:
                    print(f"\n=== Model Comparison ===")
                    print(f"Models Agree: {comparison['agreement']}")
                    print(f"Confidence Difference: {comparison['confidence_difference']:.4f}")
            except:
                print("\nModel comparison not available")
        else:
            print(f"Error: {result['error']}")
    else:
        print(f"Test image not found: {test_image_path}")
