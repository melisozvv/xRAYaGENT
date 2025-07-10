"""
Chest X-ray Anatomy Segmentation Tool
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ChestXrayAnatomySegmentation:
    """Tool for segmenting anatomical structures in chest X-rays"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.anatomy_classes = []
        
    def load_model(self, model_name: str = "anatomy_segmentation"):
        """Load the anatomy segmentation model"""
        try:
            # This would normally load a real segmentation model
            # For demonstration, we'll simulate the model loading
            logger.info(f"Loading anatomy segmentation model: {model_name}")
            
            # Simulated anatomy classes that would be detected
            self.anatomy_classes = [
                "Heart", "Left Lung", "Right Lung", "Spine", "Ribs", 
                "Diaphragm", "Mediastinum", "Trachea", "Clavicles", "Aortic Arch"
            ]
            
            logger.info(f"Anatomy segmentation model loaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load anatomy segmentation model: {e}")
            return False
    
    def _generate_mask(self, image_size: tuple, structure_name: str, bbox: List[int]) -> np.ndarray:
        """Generate a simulated mask for an anatomical structure"""
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # Create a simple mask based on bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox]  # Ensure integers
        
        # Ensure coordinates are within image bounds
        height, width = image_size
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        if x1 >= x2 or y1 >= y2:
            return mask  # Return empty mask if invalid bbox
        
        if structure_name in ["Heart"]:
            # Create an ellipse-like mask for heart
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # Simple ellipse equation
                    if ((x - center_x)**2 / max(1, ((x2-x1)//2)**2) + 
                        (y - center_y)**2 / max(1, ((y2-y1)//2)**2)) <= 1:
                        mask[y, x] = 255
        
        elif structure_name in ["Left Lung", "Right Lung"]:
            # Create a more organic shape for lungs
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # Lung-like shape
                    if ((x - center_x)**2 / max(1, ((x2-x1)//2.5)**2) + 
                        (y - center_y)**2 / max(1, ((y2-y1)//2.2)**2)) <= 1:
                        mask[y, x] = 255
        
        else:
            # Simple rectangular mask for other structures
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def segment_anatomy(self, image_path: str, return_masks: bool = False) -> Dict[str, Any]:
        """
        Segment anatomical structures in chest X-ray
        
        Args:
            image_path: Path to the X-ray image
            return_masks: Whether to return/save segmentation masks
            
        Returns:
            Dictionary with segmentation results
        """
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            # Load image
            image = Image.open(image_path)
            width, height = image.size
            
            # Ensure model is loaded
            if not self.anatomy_classes:
                if not self.load_model():
                    return {"error": "Failed to load anatomy segmentation model"}
            
            # Simulate segmentation results
            segmentation_results = {
                "Heart": {
                    "area": 18500,
                    "bbox": [width//3, height//3, width//2, height//2],
                    "centroid": [width//2.2, height//2.5],
                    "confidence": 0.92
                },
                "Left Lung": {
                    "area": 45000,
                    "bbox": [width//6, height//4, width//2.5, height//1.5],
                    "centroid": [width//3.5, height//2.2],
                    "confidence": 0.88
                },
                "Right Lung": {
                    "area": 47000,
                    "bbox": [width//1.8, height//4, width//1.2, height//1.5],
                    "centroid": [width//1.5, height//2.2],
                    "confidence": 0.90
                },
                "Spine": {
                    "area": 12000,
                    "bbox": [width//2.2, height//8, width//1.8, height//1.2],
                    "centroid": [width//2, height//2],
                    "confidence": 0.85
                },
                "Ribs": {
                    "area": 8500,
                    "bbox": [width//6, height//6, width//1.2, height//1.8],
                    "centroid": [width//2, height//2.5],
                    "confidence": 0.78
                }
            }
            
            result = {
                "anatomical_structures": segmentation_results,
                "detected_structures": list(segmentation_results.keys()),
                "image_path": image_path,
                "image_size": [width, height],
                "tool_name": "ChestXRayAnatomySegmentation",
                "total_structures": len(segmentation_results)
            }
            
            # Generate and save masks if requested
            if return_masks:
                mask_info = self._save_masks(image_path, segmentation_results, (height, width))
                result["masks"] = mask_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anatomy segmentation: {e}")
            return {"error": str(e)}
    
    def _save_masks(self, image_path: str, segmentation_results: Dict, image_size: tuple) -> Dict[str, Any]:
        """Save mask images for each anatomical structure"""
        try:
            # Create output directory
            image_dir = os.path.dirname(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_dir = os.path.join(image_dir, f"{image_name}_masks")
            os.makedirs(mask_dir, exist_ok=True)
            
            mask_paths = {}
            
            for structure_name, structure_data in segmentation_results.items():
                try:
                    # Generate mask
                    mask = self._generate_mask(image_size, structure_name, structure_data["bbox"])
                    
                    # Save mask
                    mask_filename = f"{structure_name.lower().replace(' ', '_')}_mask.png"
                    mask_path = os.path.join(mask_dir, mask_filename)
                    
                    mask_image = Image.fromarray(mask, mode='L')
                    mask_image.save(mask_path)
                    
                    mask_paths[structure_name] = mask_path
                    
                except Exception as e:
                    logger.error(f"Error generating mask for {structure_name}: {e}")
                    continue
            
            return {
                "mask_directory": mask_dir,
                "mask_paths": mask_paths,
                "total_masks": len(mask_paths)
            }
            
        except Exception as e:
            logger.error(f"Error saving masks: {e}")
            return {"error": str(e)}
    
    def process_folder(self, folder_path: str, return_masks: bool = True) -> Dict[str, Any]:
        """
        Process all images in a folder and generate mask images
        
        Args:
            folder_path: Path to the folder containing X-ray images
            return_masks: Whether to generate and save mask images
            
        Returns:
            Dictionary with batch processing results
        """
        try:
            if not os.path.exists(folder_path):
                return {"error": f"Folder not found: {folder_path}"}
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dicom']
            image_files = []
            
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(folder_path, file))
            
            if not image_files:
                return {"error": f"No image files found in folder: {folder_path}"}
            
            # Process each image
            results = {}
            total_masks = 0
            
            for i, image_path in enumerate(image_files):
                image_name = os.path.basename(image_path)
                logger.info(f"Processing image {i+1}/{len(image_files)}: {image_name}")
                
                result = self.segment_anatomy(image_path, return_masks=return_masks)
                
                if "error" not in result:
                    results[image_name] = result
                    if return_masks and "masks" in result:
                        total_masks += result["masks"]["total_masks"]
                else:
                    results[image_name] = result
            
            # Summary
            successful_images = sum(1 for r in results.values() if "error" not in r)
            
            return {
                "batch_results": results,
                "folder_path": folder_path,
                "total_images": len(image_files),
                "successful_images": successful_images,
                "total_masks_generated": total_masks,
                "masks_generated": return_masks,
                "tool_name": "ChestXRayAnatomySegmentation"
            }
            
        except Exception as e:
            logger.error(f"Error processing folder: {e}")
            return {"error": str(e)}
    
    def calculate_clinical_measurements(self, image_path: str) -> Dict[str, Any]:
        """
        Calculate clinical measurements from segmentation
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            Dictionary with clinical measurements
        """
        try:
            # First get segmentation results
            seg_result = self.segment_anatomy(image_path)
            
            if "error" in seg_result:
                return seg_result
            
            structures = seg_result["anatomical_structures"]
            image_size = seg_result["image_size"]
            
            # Calculate cardio-thoracic ratio (CTR)
            if "Heart" in structures and "Left Lung" in structures and "Right Lung" in structures:
                heart_width = structures["Heart"]["bbox"][2] - structures["Heart"]["bbox"][0]
                
                left_lung_right = structures["Left Lung"]["bbox"][2]
                right_lung_left = structures["Right Lung"]["bbox"][0]
                thoracic_width = right_lung_left - left_lung_right + heart_width
                
                ctr = heart_width / thoracic_width if thoracic_width > 0 else 0
            else:
                ctr = 0.5  # Default/simulated value
            
            # Calculate spine-heart distance
            if "Heart" in structures and "Spine" in structures:
                heart_center = structures["Heart"]["centroid"][0]
                spine_center = structures["Spine"]["centroid"][0]
                spine_heart_distance = abs(heart_center - spine_center)
            else:
                spine_heart_distance = 50  # Default/simulated value
            
            # Calculate lung areas
            left_lung_area = structures.get("Left Lung", {}).get("area", 0)
            right_lung_area = structures.get("Right Lung", {}).get("area", 0)
            total_lung_area = left_lung_area + right_lung_area
            
            return {
                "clinical_measurements": {
                    "cardio_thoracic_ratio": round(ctr, 3),
                    "spine_heart_distance": round(spine_heart_distance, 1),
                    "left_lung_area": left_lung_area,
                    "right_lung_area": right_lung_area,
                    "total_lung_area": total_lung_area,
                    "lung_area_ratio": round(left_lung_area / right_lung_area, 3) if right_lung_area > 0 else 0
                },
                "image_path": image_path,
                "tool_name": "ChestXRayAnatomySegmentation",
                "measurement_units": {
                    "cardio_thoracic_ratio": "ratio",
                    "spine_heart_distance": "pixels",
                    "lung_areas": "pixels²"
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating clinical measurements: {e}")
            return {"error": str(e)}
    
    def analyze_structure_positions(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze anatomical structure positions and relationships
        
        Args:
            image_path: Path to the X-ray image
            
        Returns:
            Dictionary with position analysis
        """
        try:
            # Get segmentation results
            seg_result = self.segment_anatomy(image_path)
            
            if "error" in seg_result:
                return seg_result
            
            structures = seg_result["anatomical_structures"]
            image_size = seg_result["image_size"]
            
            position_analysis = {}
            
            # Check heart position
            if "Heart" in structures:
                heart_center = structures["Heart"]["centroid"]
                image_center = [image_size[0]//2, image_size[1]//2]
                
                heart_offset_x = heart_center[0] - image_center[0]
                heart_offset_y = heart_center[1] - image_center[1]
                
                position_analysis["heart_position"] = {
                    "center": heart_center,
                    "offset_from_center": [heart_offset_x, heart_offset_y],
                    "position_description": "Normal" if abs(heart_offset_x) < 20 else "Shifted"
                }
            
            # Check lung symmetry
            if "Left Lung" in structures and "Right Lung" in structures:
                left_area = structures["Left Lung"]["area"]
                right_area = structures["Right Lung"]["area"]
                
                asymmetry_ratio = abs(left_area - right_area) / max(left_area, right_area)
                
                position_analysis["lung_symmetry"] = {
                    "left_area": left_area,
                    "right_area": right_area,
                    "asymmetry_ratio": round(asymmetry_ratio, 3),
                    "symmetry_description": "Symmetric" if asymmetry_ratio < 0.1 else "Asymmetric"
                }
            
            # Check spine alignment
            if "Spine" in structures:
                spine_center = structures["Spine"]["centroid"]
                image_center_x = image_size[0] // 2
                
                spine_deviation = abs(spine_center[0] - image_center_x)
                
                position_analysis["spine_alignment"] = {
                    "center": spine_center,
                    "deviation": spine_deviation,
                    "alignment_description": "Aligned" if spine_deviation < 15 else "Deviated"
                }
            
            return {
                "position_analysis": position_analysis,
                "image_path": image_path,
                "tool_name": "ChestXRayAnatomySegmentation"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing structure positions: {e}")
            return {"error": str(e)}

# Standalone testing functions
def test_anatomy_segmentation():
    """Test function for anatomy segmentation"""
    print("Testing Chest X-ray Anatomy Segmentation...")
    
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
    segmenter = ChestXrayAnatomySegmentation()
    
    # Test segmentation
    result = segmenter.segment_anatomy(image_path)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print("✅ Anatomy segmentation test passed")
    print(f"Detected structures: {result['detected_structures']}")
    
    # Test with mask generation
    mask_result = segmenter.segment_anatomy(image_path, return_masks=True)
    if "error" not in mask_result and "masks" in mask_result:
        print(f"✅ Mask generation test passed")
        print(f"Generated {mask_result['masks']['total_masks']} masks")
        print(f"Mask directory: {mask_result['masks']['mask_directory']}")
    elif "masks" not in mask_result:
        print(f"⚠️ Mask generation test failed - no masks generated")
    else:
        print(f"❌ Mask generation test failed: {mask_result.get('error', 'Unknown error')}")
    
    # Test clinical measurements
    measurements = segmenter.calculate_clinical_measurements(image_path)
    if "error" not in measurements:
        print(f"CTR: {measurements['clinical_measurements']['cardio_thoracic_ratio']}")
    
    # Test folder processing if data folder exists
    data_folder = "../../data"
    if os.path.exists(data_folder):
        print("\n🔄 Testing folder processing...")
        folder_result = segmenter.process_folder(data_folder, return_masks=True)
        if "error" not in folder_result:
            print(f"✅ Folder processing test passed")
            print(f"Processed {folder_result['successful_images']}/{folder_result['total_images']} images")
            print(f"Generated {folder_result['total_masks_generated']} total masks")
        else:
            print(f"⚠️ Folder processing test failed: {folder_result['error']}")
    
    return True

if __name__ == "__main__":
    test_anatomy_segmentation() 