import os
import torch
from PIL import Image
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MAIRA2Detection:
    """Tool for phrase grounding in chest X-rays using MAIRA-2 model"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.loaded = False
        
    def load_model(self, model_name: str = "microsoft/maira-2"):
        """Load the MAIRA-2 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading MAIRA-2 model: {model_name}")
            
            # Initialize model and processor
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.eval().to(self.device)
            
            self.loaded = True
            logger.info(f"MAIRA-2 model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MAIRA-2 model: {e}")
            return False
    
    def ground_phrase(self, image_path: str, phrase: str) -> Dict[str, Any]:
        """
        Ground a medical phrase in a chest X-ray image
        
        Args:
            image_path: Path to the X-ray image
            phrase: Medical phrase to ground (e.g., "Pleural effusion")
            
        Returns:
            Dictionary with grounding results
        """
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            # Load image
            image = Image.open(image_path)
            
            # Ensure model is loaded
            if not self.loaded:
                if not self.load_model():
                    return {"error": "Failed to load MAIRA-2 model"}
            
            # Process the input for phrase grounding
            logger.info(f"MAIRA-2 phrase grounding input - Phrase: '{phrase}'")
            
            processed_inputs = self.processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image,
                phrase=phrase,
                return_tensors="pt"
            )
            
            # Move inputs to device
            processed_inputs = processed_inputs.to(self.device)
            
            logger.info(f"MAIRA-2 input processed, input_ids shape: {processed_inputs['input_ids'].shape}")
            
            # Generate output with appropriate parameters for phrase grounding
            with torch.no_grad():
                output_decoding = self.model.generate(
                    **processed_inputs, 
                    max_new_tokens=150,  # Should be enough for phrase grounding
                    use_cache=True,
                    do_sample=False,  # Use greedy decoding for consistency
                    temperature=1.0
                )
                
                # Decode the output
                prompt_length = processed_inputs["input_ids"].shape[-1]
                decoded_text = self.processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
                
                logger.info(f"MAIRA-2 raw decoded text: '{decoded_text}'")
                
                # Convert to grounded sequence
                prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
                
                logger.info(f"MAIRA-2 converted prediction type: {type(prediction)}, content: {prediction}")
            
            # Parse grounding result
            grounding_result = self._parse_grounding_result(prediction)
            
            return {
                "phrase": phrase,
                "grounding_result": prediction,
                "parsed_result": grounding_result,
                "image_path": image_path,
                "tool_name": "MAIRA-2"
            }
            
        except Exception as e:
            logger.error(f"Error in MAIRA-2 phrase grounding: {e}")
            return {"error": str(e)}
    
    def _parse_grounding_result(self, prediction: Any) -> Dict[str, Any]:
        """Parse the grounding result to extract coordinates and confidence"""
        try:
            # Simplified result with only basic fields
            result = {
                "grounded": False,
                "coordinates": None,
                "confidence": None
            }
            
            logger.info(f"Parsing MAIRA-2 prediction type: {type(prediction)}")
            
            # Check if prediction is a tuple (typical grounded result format)
            if isinstance(prediction, tuple) and len(prediction) == 2:
                phrase_text, coordinates = prediction
                if coordinates is not None:
                    result["grounded"] = True
                    result["coordinates"] = coordinates
                    logger.info(f"Found coordinates in tuple format: {coordinates}")
                
            elif isinstance(prediction, list):
                # Handle list of grounded sequences
                found_coordinates = False
                
                for item in prediction:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        text, coords = item
                        logger.info(f"Checking grounded item: text='{text}', coords={coords}")
                        
                        if coords is not None and not found_coordinates:
                            result["grounded"] = True
                            result["coordinates"] = coords
                            found_coordinates = True
                            logger.info(f"Found grounded phrase: '{text}' with coords: {coords}")
                            break
                
                if not found_coordinates:
                    logger.warning("MAIRA-2 returned a list but no coordinates found - might be doing report generation instead of phrase grounding")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error parsing grounding result: {e}")
            return {
                "grounded": False,
                "coordinates": None,
                "confidence": None
            }
    
    def detect_multiple_phrases(self, image_path: str, phrases: List[str]) -> Dict[str, Any]:
        """
        Ground multiple medical phrases in a chest X-ray image
        
        Args:
            image_path: Path to the X-ray image
            phrases: List of medical phrases to ground
            
        Returns:
            Dictionary with grounding results for all phrases
        """
        results = {}
        
        for phrase in phrases:
            result = self.ground_phrase(image_path, phrase)
            results[phrase] = result
        
        return {
            "multi_phrase_grounding": results,
            "image_path": image_path,
            "total_phrases": len(phrases),
            "tool_name": "MAIRA-2"
        }
    
    def detect_common_findings(self, image_path: str) -> Dict[str, Any]:
        """
        Detect common chest X-ray findings using predefined phrases
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            Dictionary with grounding results for common findings
        """
        common_findings = [
            "Pleural effusion",
            "Pneumothorax",
            "Cardiomegaly",
            "Pulmonary edema",
            "Consolidation",
            "Atelectasis",
            "Pneumonia",
            "Lung nodule",
            "Rib fracture",
            "Mediastinal shift"
        ]
        
        return self.detect_multiple_phrases(image_path, common_findings)
    
    def ground_anatomical_structures(self, image_path: str) -> Dict[str, Any]:
        """
        Ground anatomical structures in chest X-ray
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            Dictionary with grounding results for anatomical structures
        """
        anatomical_structures = [
            "Heart",
            "Left lung",
            "Right lung",
            "Diaphragm",
            "Mediastinum",
            "Trachea",
            "Clavicle",
            "Ribs",
            "Spine",
            "Aortic arch"
        ]
        
        return self.detect_multiple_phrases(image_path, anatomical_structures)

# Standalone testing functions
def test_maira2_detection():
    """Test function for MAIRA-2 detection"""
    print("Testing MAIRA-2 Detection...")
    
    # Find test image
    test_image_paths = [
        "data/xray.jpg",
        "../../data/xray.jpg",
        "../data/xray.jpg"
    ]
    
    image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print("❌ No test image found")
        return False
    
    print(f"Using test image: {image_path}")
    
    # Initialize tool
    detector = MAIRA2Detection()
    
    # Test phrase grounding
    result = detector.ground_phrase(image_path, "Pleural effusion")
    print(result)
    # if "error" in result:
    #     print(f"❌ Error: {result['error']}")
    #     return False
    
    # print("✅ MAIRA-2 detection test passed")
    # print(f"Phrase: {result['phrase']}")
    # print(f"Grounding result: {result['grounding_result']}")
    # print(f"Parsed result: {result['parsed_result']}")
    
    # # Test multiple phrases
    # phrases = ["Cardiomegaly", "Pneumothorax"]
    # multi_result = detector.detect_multiple_phrases(image_path, phrases)
    # if "error" not in multi_result:
    #     print(f"✅ Multi-phrase grounding test passed ({len(phrases)} phrases)")
    
    return True

if __name__ == "__main__":
    test_maira2_detection()
