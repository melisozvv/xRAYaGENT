"""
Chest X-ray Anatomy Segmentation Tool
"""

import os
import subprocess
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import logging
import json
import glob

logger = logging.getLogger(__name__)

class ChestXrayAnatomySegmentation:
    """Tool for segmenting anatomical structures in chest X-rays using cxas_segment"""
    
    def __init__(self):
        self.output_dir = None
        
    def check_cxas_availability(self) -> bool:
        """Check if cxas_segment command is available"""
        try:
            result = subprocess.run(['cxas_segment', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def segment_anatomy(self, image_path: str, return_masks: bool = None, output_dir: str = None, 
                       output_type: str = "png", device: str = "cpu") -> Dict[str, Any]:
        """
        Segment anatomical structures in chest X-ray using cxas_segment
        
        Args:
            image_path: Path to the X-ray image
            return_masks: Whether to return/save segmentation masks (for backward compatibility)
            output_dir: Output directory for segmentation results
            output_type: Output format (png, nii, etc.)
            device: Device to run on (cpu, gpu, cuda, 0, 1, etc.)
            
        Returns:
            Dictionary with segmentation results
        """
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            # Check if cxas_segment is available
            if not self.check_cxas_availability():
                return {"error": "cxas_segment command not found. Please install ChestXRayAnatomySegmentation package."}
            
            # Set default output directory
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(image_path), "output")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            
            # Prepare command
            cmd = [
                "cxas_segment",
                "-i", image_path,
                "-o", output_dir,
                "-ot", output_type,
                "-g", str(device)
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the segmentation command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                error_msg = f"cxas_segment failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr}"
                return {"error": error_msg}
            
            # Parse results
            segmentation_results = self._parse_segmentation_output(image_path, output_dir)
            
            response = {
                "anatomical_structures": segmentation_results.get("structures", {}),
                "detected_structures": segmentation_results.get("detected_structures", []),
                "image_path": image_path,
                "output_directory": output_dir,
                "tool_name": "ChestXRayAnatomySegmentation",
                "command_output": result.stdout,
                "total_structures": len(segmentation_results.get("detected_structures", []))
            }
            
            # Add masks info if requested (backward compatibility)
            if return_masks is True or return_masks is None:
                response["masks"] = segmentation_results.get("masks", {})
            
            return response
            
        except subprocess.TimeoutExpired:
            return {"error": "cxas_segment command timed out"}
        except Exception as e:
            logger.error(f"Error in anatomy segmentation: {e}")
            return {"error": str(e)}
    
    def _parse_segmentation_output(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Parse the output files from cxas_segment"""
        try:
            results = {
                "structures": {},
                "detected_structures": [],
                "masks": {}
            }
            
            # Common anatomy labels that cxas_segment might produce
            anatomy_labels = {
                "heart": "Heart",
                "left_lung": "Left Lung", 
                "right_lung": "Right Lung",
                "spine": "Spine",
                "ribs": "Ribs",
                "trachea": "Trachea",
                "clavicles": "Clavicles",
                "diaphragm": "Diaphragm"
            }
            
            # Find output files
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Look for mask files
            mask_files = []
            for pattern in [f"{output_dir}/*{image_basename}*", f"{output_dir}/*"]:
                mask_files.extend(glob.glob(pattern))
            
            # Filter for image files
            mask_files = [f for f in mask_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))]
            
            # Process each mask file
            for mask_file in mask_files:
                try:
                    mask_basename = os.path.basename(mask_file)
                    
                    # Try to identify the anatomical structure
                    structure_name = None
                    for key, label in anatomy_labels.items():
                        if key in mask_basename.lower():
                            structure_name = label
                            break
                    
                    if not structure_name:
                        # Generic naming if we can't identify the structure
                        structure_name = f"Structure_{len(results['structures'])}"
                    
                    # Load mask to get basic info
                    if mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        mask_image = Image.open(mask_file)
                        mask_array = np.array(mask_image)
                        
                        # Get mask statistics
                        if len(mask_array.shape) == 3:
                            mask_array = mask_array[:,:,0]  # Take first channel
                        
                        non_zero_pixels = np.count_nonzero(mask_array)
                        total_pixels = mask_array.size
                        
                        # Find bounding box
                        if non_zero_pixels > 0:
                            coords = np.where(mask_array > 0)
                            y_min, y_max = coords[0].min(), coords[0].max()
                            x_min, x_max = coords[1].min(), coords[1].max()
                            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                            centroid = [int((x_min + x_max) / 2), int((y_min + y_max) / 2)]
                        else:
                            bbox = [0, 0, 0, 0]
                            centroid = [0, 0]
                        
                        results["structures"][structure_name] = {
                            "area": int(non_zero_pixels),
                            "bbox": bbox,
                            "centroid": centroid,
                            "mask_file": mask_file,
                            "confidence": 1.0  # cxas_segment doesn't provide confidence scores
                        }
                        
                        results["detected_structures"].append(structure_name)
                        results["masks"][structure_name] = mask_file
                        
                except Exception as e:
                    logger.warning(f"Error processing mask file {mask_file}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing segmentation output: {e}")
            return {"structures": {}, "detected_structures": [], "masks": {}}
    
    def process_folder(self, folder_path: str, return_masks: bool = True, output_dir: str = None, 
                      output_type: str = "png", device: str = "cpu") -> Dict[str, Any]:
        """
        Process all images in a folder using cxas_segment
        
        Args:
            folder_path: Path to the folder containing X-ray images
            return_masks: Whether to generate and save mask images (for backward compatibility)
            output_dir: Output directory for all results
            output_type: Output format (png, nii, etc.)
            device: Device to run on (cpu, gpu, etc.)
            
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
            
            # Set default output directory
            if output_dir is None:
                output_dir = os.path.join(folder_path, "segmentation_output")
            
            # Process each image
            results = {}
            total_structures = 0
            
            for i, image_path in enumerate(image_files):
                image_name = os.path.basename(image_path)
                logger.info(f"Processing image {i+1}/{len(image_files)}: {image_name}")
                
                # Create subdirectory for this image
                image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
                
                result = self.segment_anatomy(image_path, return_masks=return_masks, 
                                            output_dir=image_output_dir, output_type=output_type, device=device)
                
                if "error" not in result:
                    results[image_name] = result
                    total_structures += result.get("total_structures", 0)
                else:
                    results[image_name] = result
            
            # Summary
            successful_images = sum(1 for r in results.values() if "error" not in r)
            
            return {
                "batch_results": results,
                "folder_path": folder_path,
                "output_directory": output_dir,
                "total_images": len(image_files),
                "successful_images": successful_images,
                "total_structures_detected": total_structures,
                "tool_name": "ChestXRayAnatomySegmentation"
            }
            
        except Exception as e:
            logger.error(f"Error processing folder: {e}")
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
            # Get segmentation results first
            seg_result = self.segment_anatomy(image_path, return_masks=False)
            
            if "error" in seg_result:
                return seg_result
            
            structures = seg_result["anatomical_structures"]
            
            # Get image dimensions
            image = Image.open(image_path)
            image_size = [image.width, image.height]
            
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
    
    def calculate_clinical_measurements(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Calculate clinical measurements from cxas_segment segmentation
        
        Args:
            image_path: Path to the X-ray image
            output_dir: Output directory for segmentation
            
        Returns:
            Dictionary with clinical measurements
        """
        try:
            # First get segmentation results
            seg_result = self.segment_anatomy(image_path, return_masks=False, output_dir=output_dir)
            
            if "error" in seg_result:
                return seg_result
            
            structures = seg_result["anatomical_structures"]
            
            # Calculate cardio-thoracic ratio (CTR)
            ctr = 0.5  # Default value
            if "Heart" in structures and ("Left Lung" in structures or "Right Lung" in structures):
                heart_bbox = structures["Heart"]["bbox"]
                heart_width = heart_bbox[2] - heart_bbox[0]
                
                # Try to calculate thoracic width
                thoracic_width = heart_width * 2  # Rough estimate
                if "Left Lung" in structures and "Right Lung" in structures:
                    left_lung_bbox = structures["Left Lung"]["bbox"]
                    right_lung_bbox = structures["Right Lung"]["bbox"]
                    thoracic_width = right_lung_bbox[2] - left_lung_bbox[0]
                
                ctr = heart_width / thoracic_width if thoracic_width > 0 else 0.5
            
            # Calculate spine-heart distance
            spine_heart_distance = 50  # Default value
            if "Heart" in structures and "Spine" in structures:
                heart_center = structures["Heart"]["centroid"][0]
                spine_center = structures["Spine"]["centroid"][0]
                spine_heart_distance = abs(heart_center - spine_center)
            
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

# Standalone testing functions
def test_anatomy_segmentation():
    """Test function for anatomy segmentation using cxas_segment"""
    print("Testing Chest X-ray Anatomy Segmentation with cxas_segment...")
    
    # Find test image
    test_image_paths = [
        "../../data/xray.jpg",
        "../data/xray.jpg",
        "data/xray.jpg"
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
    
    # Check if cxas_segment is available
    if not segmenter.check_cxas_availability():
        print("❌ cxas_segment command not found. Please install ChestXRayAnatomySegmentation package.")
        return False
    
    # Test segmentation
    print(f"🔄 Running cxas_segment on {image_path}...")
    output_dir = os.path.join(os.path.dirname(image_path), "output")
    result = segmenter.segment_anatomy(image_path, output_dir=output_dir, output_type="png", device="cpu")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print("✅ Anatomy segmentation test passed")
    print(f"Detected structures: {result['detected_structures']}")
    print(f"Output directory: {result['output_directory']}")
    
    if result.get('masks'):
        print(f"Generated mask files:")
        for structure, mask_path in result['masks'].items():
            print(f"  - {structure}: {mask_path}")
    
    # Test clinical measurements
    print("\n🔄 Testing clinical measurements...")
    measurements = segmenter.calculate_clinical_measurements(image_path, output_dir=output_dir)
    if "error" not in measurements:
        print("✅ Clinical measurements calculated:")
        for measure, value in measurements['clinical_measurements'].items():
            print(f"  - {measure}: {value}")
    else:
        print(f"⚠️ Clinical measurements failed: {measurements['error']}")
    
    return True

if __name__ == "__main__":
    test_anatomy_segmentation() 