"""
ChestXRayAnatomySegmentation Tool - Anatomical Segmentation and Feature Extraction
"""

import os
import subprocess
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Optional
import logging
import tempfile
import shutil
import torch
import argparse

logger = logging.getLogger(__name__)

class ChestXrayAnatomySegmentation:
    """ChestXRayAnatomySegmentation tool for chest X-ray anatomical segmentation"""
    
    def __init__(self):
        self.output_dir = None
        self.temp_dir = None
        self._patch_torch_for_cxas()
        
    def _patch_torch_for_cxas(self):
        """Apply PyTorch patch for CXAS compatibility with PyTorch 2.6+"""
        try:
            # Add argparse.Namespace to safe globals to fix CXAS loading issue
            torch.serialization.add_safe_globals([argparse.Namespace])
            logger.info("Applied PyTorch patch for CXAS compatibility")
        except Exception as e:
            logger.warning(f"Could not apply PyTorch patch: {e}")
    
    def _create_temp_output_dir(self) -> str:
        """Create a temporary directory for output"""
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir
    
    def _run_cxas_segmentation(self, image_path: str, output_dir: str, 
                             output_type: str = "png", gpus: str = "cpu") -> Dict[str, Any]:
        """Run CXAS segmentation using Python API instead of command line"""
        try:
            # Import CXAS after applying the patch
            from cxas.segmentor import CXAS
            
            logger.info(f"Running CXAS segmentation on {image_path}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize CXAS model
            model = CXAS(gpus=gpus)
            
            # Run segmentation
            result = model.segment(image_path, output_dir, output_type=output_type)
            
            return {"success": True, "result": result}
            
        except ImportError as e:
            logger.error(f"CXAS not installed or not available: {e}")
            return {"success": False, "error": f"CXAS not available: {e}"}
        except Exception as e:
            logger.error(f"CXAS segmentation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def segment_anatomy(self, image_path: str, output_type: str = "png", 
                       model: Optional[str] = None, 
                       gpus: str = "cpu") -> Dict[str, Any]:
        """
        Segment anatomical structures in chest X-ray using CXAS
        
        Args:
            image_path: Path to the input chest X-ray image
            output_type: Output format for segmentation masks
            model: Segmentation model to use (optional, uses CXAS default)
            gpus: GPU device to use ('cpu' for CPU-only)
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Create temporary output directory
            output_dir = self._create_temp_output_dir()
            
            # Run CXAS segmentation using Python API
            result = self._run_cxas_segmentation(image_path, output_dir, output_type, gpus)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "segmentation_masks": [],
                    "anatomical_structures": []
                }
            
            # Collect output files
            segmentation_masks = []
            anatomical_structures = []
            
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    if file.endswith(f".{output_type}"):
                        full_path = os.path.join(output_dir, file)
                        segmentation_masks.append(full_path)
                        # Extract structure name from filename
                        structure_name = file.replace(f".{output_type}", "")
                        anatomical_structures.append(structure_name)
            
            return {
                "success": True,
                "segmentation_masks": segmentation_masks,
                "anatomical_structures": anatomical_structures,
                "output_directory": output_dir,
                "model_used": "CXAS_default",
                "output_format": output_type
            }
            
        except Exception as e:
            logger.error(f"Error in anatomy segmentation: {e}")
            return {
                "success": False,
                "error": str(e),
                "segmentation_masks": [],
                "anatomical_structures": []
            }
    
    def extract_features(self, image_path: str, features: List[str], 
                        output_type: str = "csv", gpus: str = "cpu",
                        store_seg: bool = False) -> Dict[str, Any]:
        """
        Extract anatomical features from chest X-ray
        
        Args:
            image_path: Path to the input chest X-ray image
            features: List of features to extract (CTR, SCD, etc.)
            output_type: Output format for features
            gpus: GPU device to use ('cpu' for CPU-only)
            store_seg: Whether to store segmentation masks
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Create temporary output directory
            output_dir = self._create_temp_output_dir()
            
            # Build command
            command = [
                "cxas_feat_extract",
                "-i", image_path,
                "-o", output_dir,
                "-ot", output_type,
                "-g", gpus
            ]
            
            # Add features
            for feature in features:
                command.extend(["-f", feature])
            
            # Add store segmentation flag if requested
            if store_seg:
                command.append("-s")
            
            # Run command
            result = self._run_cxas_command(command)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "clinical_measurements": {},
                    "feature_file": None
                }
            
            # Look for output files
            feature_file = None
            clinical_measurements = {}
            
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    if file.endswith(f".{output_type}"):
                        feature_file = os.path.join(output_dir, file)
                        
                        # If it's a CSV file, try to read the measurements
                        if output_type == "csv" and os.path.exists(feature_file):
                            try:
                                import pandas as pd
                                df = pd.read_csv(feature_file)
                                if not df.empty:
                                    # Convert to dictionary
                                    clinical_measurements = df.iloc[0].to_dict()
                            except:
                                # Fallback if pandas is not available
                                pass
            
            return {
                "success": True,
                "feature_file": feature_file,
                "clinical_measurements": clinical_measurements,
                "features_extracted": features,
                "output_directory": output_dir,
                "output_format": output_type
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "clinical_measurements": {},
                "feature_file": None
            }
    
    def analyze_structure_positions(self, image_path: str, 
                                  structures: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze the positions of anatomical structures
        
        Args:
            image_path: Path to the input chest X-ray image
            structures: List of specific structures to analyze (optional)
            
        Returns:
            Dictionary containing structure position analysis
        """
        try:
            # First do segmentation to get structure positions
            seg_result = self.segment_anatomy(image_path, output_type="png")
            
            if not seg_result["success"]:
                return {
                    "success": False,
                    "error": seg_result["error"],
                    "structure_positions": {}
                }
            
            structure_positions = {}
            
            # Analyze each segmented structure
            for i, mask_path in enumerate(seg_result["segmentation_masks"]):
                if os.path.exists(mask_path):
                    structure_name = seg_result["anatomical_structures"][i]
                    
                    # Skip if we're looking for specific structures and this isn't one
                    if structures and structure_name not in structures:
                        continue
                    
                    try:
                        # Load mask and analyze position
                        mask = Image.open(mask_path)
                        mask_array = np.array(mask)
                        
                        # Find non-zero pixels (structure pixels)
                        y_coords, x_coords = np.where(mask_array > 0)
                        
                        if len(y_coords) > 0:
                            # Calculate position statistics
                            position_info = {
                                "centroid": {
                                    "x": float(np.mean(x_coords)),
                                    "y": float(np.mean(y_coords))
                                },
                                "bounding_box": {
                                    "min_x": int(np.min(x_coords)),
                                    "max_x": int(np.max(x_coords)),
                                    "min_y": int(np.min(y_coords)),
                                    "max_y": int(np.max(y_coords))
                                },
                                "area": int(len(y_coords)),
                                "relative_position": {
                                    "x_normalized": float(np.mean(x_coords) / mask_array.shape[1]),
                                    "y_normalized": float(np.mean(y_coords) / mask_array.shape[0])
                                }
                            }
                            
                            structure_positions[structure_name] = position_info
                    
                    except Exception as e:
                        logger.warning(f"Could not analyze position for {structure_name}: {e}")
            
            return {
                "success": True,
                "structure_positions": structure_positions,
                "image_analyzed": image_path,
                "total_structures": len(structure_positions)
            }
            
        except Exception as e:
            logger.error(f"Error in structure position analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "structure_positions": {}
            }
    
    def calculate_ctr(self, image_path: str) -> Dict[str, Any]:
        """
        Calculate Cardio-Thoracic Ratio (CTR)
        
        Args:
            image_path: Path to the input chest X-ray image
            
        Returns:
            Dictionary containing CTR calculation results
        """
        try:
            # Extract CTR feature
            result = self.extract_features(
                image_path, 
                features=["CTR", "Cardio-Thoracic Ratio"], 
                output_type="csv"
            )
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "ctr_value": None
                }
            
            # Extract CTR value from clinical measurements
            ctr_value = None
            measurements = result.get("clinical_measurements", {})
            
            for key, value in measurements.items():
                if "ctr" in key.lower() or "cardio" in key.lower():
                    try:
                        ctr_value = float(value)
                        break
                    except:
                        pass
            
            return {
                "success": True,
                "ctr_value": ctr_value,
                "clinical_measurements": measurements,
                "interpretation": self._interpret_ctr(ctr_value) if ctr_value else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating CTR: {e}")
            return {
                "success": False,
                "error": str(e),
                "ctr_value": None
            }
    
    def _interpret_ctr(self, ctr_value: float) -> str:
        """Interpret CTR value clinically"""
        if ctr_value is None:
            return "Unable to interpret - no CTR value"
        
        if ctr_value <= 0.5:
            return "Normal (CTR ≤ 0.5)"
        elif ctr_value <= 0.6:
            return "Borderline enlarged (0.5 < CTR ≤ 0.6)"
        else:
            return "Enlarged (CTR > 0.6)"
    
    def create_bounding_boxes(self, image_path: str, 
                            target_structures: Optional[List[str]] = None,
                            save_visualization: bool = False,
                            visualization_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create bounding boxes for bones and organs in chest X-ray
        
        Args:
            image_path: Path to the input chest X-ray image
            target_structures: List of specific structures to create bounding boxes for.
                             If None, will create for all detected bones and organs
            save_visualization: Whether to save a visualization image with bounding boxes
            visualization_path: Path to save the visualization (optional)
            
        Returns:
            Dictionary containing bounding box information for bones and organs
        """
        try:
            # Define common bone and organ structures in chest X-rays
            bones_and_organs = {
                'bones': [
                    'ribs', 'rib', 'spine', 'vertebra', 'vertebrae', 'clavicle', 
                    'sternum', 'scapula', 'humerus', 'thoracic_spine', 'cervical_spine'
                ],
                'organs': [
                    'heart', 'lung', 'lungs', 'left_lung', 'right_lung', 'aorta',
                    'pulmonary', 'cardiac', 'mediastinum', 'diaphragm', 'trachea'
                ]
            }
            
            # First perform segmentation
            seg_result = self.segment_anatomy(image_path, output_type="png")
            
            if not seg_result["success"]:
                return {
                    "success": False,
                    "error": seg_result["error"],
                    "bounding_boxes": {},
                    "bones_detected": [],
                    "organs_detected": []
                }
            
            # Load original image for visualization
            original_image = None
            if save_visualization:
                try:
                    original_image = Image.open(image_path)
                    if original_image.mode != 'RGB':
                        original_image = original_image.convert('RGB')
                except Exception as e:
                    logger.warning(f"Could not load original image for visualization: {e}")
            
            bounding_boxes = {}
            bones_detected = []
            organs_detected = []
            
            # Process each segmented structure
            for i, mask_path in enumerate(seg_result["segmentation_masks"]):
                if not os.path.exists(mask_path):
                    continue
                    
                structure_name = seg_result["anatomical_structures"][i].lower()
                
                # Filter for target structures if specified
                if target_structures:
                    if not any(target.lower() in structure_name for target in target_structures):
                        continue
                else:
                    # Check if it's a bone or organ
                    is_bone = any(bone in structure_name for bone in bones_and_organs['bones'])
                    is_organ = any(organ in structure_name for organ in bones_and_organs['organs'])
                    
                    if not (is_bone or is_organ):
                        continue
                
                try:
                    # Load and process mask
                    mask = Image.open(mask_path)
                    mask_array = np.array(mask)
                    
                    # Find structure pixels
                    y_coords, x_coords = np.where(mask_array > 0)
                    
                    if len(y_coords) > 0:
                        # Calculate bounding box
                        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
                        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
                        
                        # Calculate additional metrics
                        width = max_x - min_x
                        height = max_y - min_y
                        area = int(len(y_coords))
                        center_x = min_x + width // 2
                        center_y = min_y + height // 2
                        
                        # Determine structure type
                        structure_type = 'unknown'
                        if any(bone in structure_name for bone in bones_and_organs['bones']):
                            structure_type = 'bone'
                            bones_detected.append(structure_name)
                        elif any(organ in structure_name for organ in bones_and_organs['organs']):
                            structure_type = 'organ'
                            organs_detected.append(structure_name)
                        
                        # Store bounding box information
                        bounding_boxes[structure_name] = {
                            "type": structure_type,
                            "bounding_box": {
                                "min_x": min_x,
                                "min_y": min_y,
                                "max_x": max_x,
                                "max_y": max_y,
                                "width": width,
                                "height": height,
                                "center_x": center_x,
                                "center_y": center_y
                            },
                            "area_pixels": area,
                            "confidence_score": 1.0,  # Could be enhanced with actual confidence if available
                            "mask_file": mask_path
                        }
                        
                        # Add to visualization if requested
                        if save_visualization and original_image:
                            try:
                                draw = ImageDraw.Draw(original_image)
                                
                                # Choose color based on type
                                color = 'red' if structure_type == 'bone' else 'blue'
                                
                                # Draw bounding box
                                draw.rectangle([min_x, min_y, max_x, max_y], 
                                             outline=color, width=2)
                                
                                # Add label
                                label = f"{structure_name} ({structure_type})"
                                draw.text((min_x, min_y - 20), label, fill=color)
                                
                            except ImportError:
                                logger.warning("PIL ImageDraw not available for visualization")
                            except Exception as e:
                                logger.warning(f"Error adding visualization for {structure_name}: {e}")
                
                except Exception as e:
                    logger.warning(f"Could not process structure {structure_name}: {e}")
            
            # Save visualization if requested
            visualization_saved = False
            if save_visualization and original_image:
                try:
                    if not visualization_path:
                        output_dir = seg_result.get("output_directory", self.temp_dir)
                        visualization_path = os.path.join(output_dir, "bounding_boxes_visualization.png")
                    
                    original_image.save(visualization_path)
                    visualization_saved = True
                    logger.info(f"Bounding box visualization saved to: {visualization_path}")
                    
                except Exception as e:
                    logger.warning(f"Could not save visualization: {e}")
                    visualization_path = None
            
            return {
                "success": True,
                "bounding_boxes": bounding_boxes,
                "bones_detected": bones_detected,
                "organs_detected": organs_detected,
                "total_structures": len(bounding_boxes),
                "image_analyzed": image_path,
                "output_directory": seg_result.get("output_directory"),
                "visualization_saved": visualization_saved,
                "visualization_path": visualization_path if visualization_saved else None,
                "structure_summary": {
                    "total_bones": len(bones_detected),
                    "total_organs": len(organs_detected),
                    "bone_list": bones_detected,
                    "organ_list": organs_detected
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating bounding boxes: {e}")
            return {
                "success": False,
                "error": str(e),
                "bounding_boxes": {},
                "bones_detected": [],
                "organs_detected": []
            }
    
    def get_structure_bounding_box(self, image_path: str, structure_name: str) -> Dict[str, Any]:
        """
        Get bounding box for a specific anatomical structure
        
        Args:
            image_path: Path to the input chest X-ray image
            structure_name: Name of the specific structure to get bounding box for
            
        Returns:
            Dictionary containing bounding box information for the specified structure
        """
        try:
            result = self.create_bounding_boxes(
                image_path, 
                target_structures=[structure_name]
            )
            
            if not result["success"]:
                return result
            
            # Find the specific structure
            structure_key = None
            for key in result["bounding_boxes"].keys():
                if structure_name.lower() in key.lower():
                    structure_key = key
                    break
            
            if structure_key:
                return {
                    "success": True,
                    "structure_name": structure_key,
                    "bounding_box": result["bounding_boxes"][structure_key],
                    "structure_type": result["bounding_boxes"][structure_key]["type"]
                }
            else:
                return {
                    "success": False,
                    "error": f"Structure '{structure_name}' not found in segmentation results",
                    "available_structures": list(result["bounding_boxes"].keys())
                }
                
        except Exception as e:
            logger.error(f"Error getting bounding box for {structure_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_bounding_boxes_folder(self, image_path: str, 
                                   output_folder: str,
                                   target_structures: Optional[List[str]] = None,
                                   include_cropped: bool = True,
                                   include_masks: bool = True,
                                   include_metadata: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive folder with bounding boxes, visualizations, and metadata
        
        Args:
            image_path: Path to the input chest X-ray image
            output_folder: Path to the output folder to create
            target_structures: List of specific structures to process (optional)
            include_cropped: Whether to include cropped images of each structure
            include_masks: Whether to include individual mask files
            include_metadata: Whether to include JSON and CSV metadata files
            
        Returns:
            Dictionary containing information about created files and folders
        """
        try:
            # Create output folder structure
            os.makedirs(output_folder, exist_ok=True)
            
            # Create subfolders
            subfolders = {
                'visualizations': os.path.join(output_folder, 'visualizations'),
                'cropped_images': os.path.join(output_folder, 'cropped_images'),
                'masks': os.path.join(output_folder, 'masks'),
                'metadata': os.path.join(output_folder, 'metadata')
            }
            
            for folder in subfolders.values():
                os.makedirs(folder, exist_ok=True)
            
            # Get bounding boxes
            bbox_result = self.create_bounding_boxes(
                image_path, 
                target_structures=target_structures,
                save_visualization=True
            )
            
            if not bbox_result["success"]:
                return {
                    "success": False,
                    "error": bbox_result["error"],
                    "output_folder": output_folder
                }
            
            # Load original image
            original_image = Image.open(image_path)
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            created_files = {
                'visualizations': [],
                'cropped_images': [],
                'masks': [],
                'metadata': [],
                'readme': []
            }
            
            # Create main visualization with all bounding boxes
            main_viz = original_image.copy()
            draw = ImageDraw.Draw(main_viz)
            
            # Process each detected structure
            for structure_name, bbox_info in bbox_result["bounding_boxes"].items():
                bbox = bbox_info["bounding_box"]
                structure_type = bbox_info["type"]
                
                # Choose color based on type
                color = 'red' if structure_type == 'bone' else 'blue'
                
                # Draw bounding box on main visualization
                draw.rectangle([bbox["min_x"], bbox["min_y"], bbox["max_x"], bbox["max_y"]], 
                             outline=color, width=3)
                
                # Add label
                label = f"{structure_name} ({structure_type})"
                draw.text((bbox["min_x"], bbox["min_y"] - 25), label, fill=color)
                
                # Create individual visualization for this structure
                individual_viz = original_image.copy()
                individual_draw = ImageDraw.Draw(individual_viz)
                individual_draw.rectangle([bbox["min_x"], bbox["min_y"], bbox["max_x"], bbox["max_y"]], 
                                        outline=color, width=3)
                individual_draw.text((bbox["min_x"], bbox["min_y"] - 25), label, fill=color)
                
                # Save individual visualization
                individual_viz_path = os.path.join(subfolders['visualizations'], f"{structure_name}_bbox.png")
                individual_viz.save(individual_viz_path)
                created_files['visualizations'].append(individual_viz_path)
                
                # Create cropped image if requested
                if include_cropped:
                    try:
                        # Add some padding to the crop
                        padding = 10
                        crop_box = (
                            max(0, bbox["min_x"] - padding),
                            max(0, bbox["min_y"] - padding),
                            min(original_image.width, bbox["max_x"] + padding),
                            min(original_image.height, bbox["max_y"] + padding)
                        )
                        
                        cropped_image = original_image.crop(crop_box)
                        cropped_path = os.path.join(subfolders['cropped_images'], f"{structure_name}_cropped.png")
                        cropped_image.save(cropped_path)
                        created_files['cropped_images'].append(cropped_path)
                        
                    except Exception as e:
                        logger.warning(f"Could not create cropped image for {structure_name}: {e}")
                
                # Copy mask file if requested and available
                if include_masks and "mask_file" in bbox_info:
                    try:
                        mask_source = bbox_info["mask_file"]
                        if os.path.exists(mask_source):
                            mask_dest = os.path.join(subfolders['masks'], f"{structure_name}_mask.png")
                            shutil.copy2(mask_source, mask_dest)
                            created_files['masks'].append(mask_dest)
                    except Exception as e:
                        logger.warning(f"Could not copy mask for {structure_name}: {e}")
            
            # Save main visualization with all bounding boxes
            main_viz_path = os.path.join(subfolders['visualizations'], "all_structures_bbox.png")
            main_viz.save(main_viz_path)
            created_files['visualizations'].append(main_viz_path)
            
            # Create metadata files if requested
            if include_metadata:
                try:
                    # Save complete bounding box data as JSON
                    metadata = {
                        "source_image": os.path.basename(image_path),
                        "analysis_timestamp": str(json.dumps(bbox_result, indent=2)),
                        "total_structures_detected": bbox_result["total_structures"],
                        "bones_detected": bbox_result["bones_detected"],
                        "organs_detected": bbox_result["organs_detected"],
                        "bounding_boxes": bbox_result["bounding_boxes"]
                    }
                    
                    metadata_path = os.path.join(subfolders['metadata'], "bounding_boxes_data.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    created_files['metadata'].append(metadata_path)
                    
                    # Create a summary CSV file
                    summary_data = []
                    for structure_name, bbox_info in bbox_result["bounding_boxes"].items():
                        bbox = bbox_info["bounding_box"]
                        summary_data.append({
                            "structure_name": structure_name,
                            "type": bbox_info["type"],
                            "min_x": bbox["min_x"],
                            "min_y": bbox["min_y"],
                            "max_x": bbox["max_x"],
                            "max_y": bbox["max_y"],
                            "width": bbox["width"],
                            "height": bbox["height"],
                            "center_x": bbox["center_x"],
                            "center_y": bbox["center_y"],
                            "area_pixels": bbox_info["area_pixels"]
                        })
                    
                    # Save as CSV (simple format without pandas dependency)
                    csv_path = os.path.join(subfolders['metadata'], "bounding_boxes_summary.csv")
                    with open(csv_path, 'w') as f:
                        if summary_data:
                            # Write header
                            headers = list(summary_data[0].keys())
                            f.write(','.join(headers) + '\n')
                            
                            # Write data
                            for row in summary_data:
                                values = [str(row[header]) for header in headers]
                                f.write(','.join(values) + '\n')
                    
                    created_files['metadata'].append(csv_path)
                    
                except Exception as e:
                    logger.warning(f"Could not create metadata files: {e}")
            
            # Create README file
            readme_path = os.path.join(output_folder, "README.txt")
            with open(readme_path, 'w') as f:
                f.write("Chest X-Ray Anatomy Bounding Boxes Analysis\n")
                f.write("=" * 45 + "\n\n")
                f.write(f"Source Image: {os.path.basename(image_path)}\n")
                f.write(f"Total Structures Detected: {bbox_result['total_structures']}\n")
                f.write(f"Bones Detected: {len(bbox_result['bones_detected'])}\n")
                f.write(f"Organs Detected: {len(bbox_result['organs_detected'])}\n\n")
                
                f.write("Folder Structure:\n")
                f.write("- visualizations/: Images with bounding boxes drawn\n")
                f.write("- cropped_images/: Individual cropped regions of each structure\n")
                f.write("- masks/: Segmentation masks for each structure\n")
                f.write("- metadata/: JSON and CSV files with bounding box coordinates\n\n")
                
                f.write("Detected Structures:\n")
                for structure_name, bbox_info in bbox_result["bounding_boxes"].items():
                    f.write(f"- {structure_name} ({bbox_info['type']})\n")
            
            created_files['readme'].append(readme_path)
            
            return {
                "success": True,
                "output_folder": output_folder,
                "created_files": created_files,
                "total_files_created": sum(len(files) for files in created_files.values() if isinstance(files, list)) + 1,
                "structure_summary": bbox_result["structure_summary"],
                "folder_structure": subfolders
            }
            
        except Exception as e:
            logger.error(f"Error creating bounding boxes folder: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_folder": output_folder
            }
    
    def cleanup(self):
        """Clean up temporary directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
    
    def __del__(self):
        """Destructor to clean up temporary files"""
        self.cleanup() 


def test_anatomy_segmentation():
    """
    Test function to demonstrate anatomy segmentation and bounding box creation
    Uses xray.jpg and saves results to xray_masks folder
    """
    print("Testing Chest X-Ray Anatomy Segmentation and Bounding Box Creation")
    print("=" * 65)
    
    try:
        # Initialize the segmentation tool
        anatomy_tool = ChestXrayAnatomySegmentation()
        
        # Define input and output paths
        input_image = "../../data/xray.jpg"
        output_folder = "../../data/xray_masks"
        
        # Check if input image exists
        import os
        if not os.path.exists(input_image):
            print(f"❌ Error: Input image '{input_image}' not found!")
            print("Please make sure xray.jpg is in the current directory.")
            return
        
        print(f"📁 Input image: {input_image}")
        print(f"📁 Output folder: {output_folder}")
        print("\n🔍 Starting anatomy segmentation and bounding box analysis...")
        
        # Create comprehensive bounding boxes analysis
        result = anatomy_tool.create_bounding_boxes_folder(
            image_path=input_image,
            output_folder=output_folder,
            target_structures=None,  # Analyze all detected structures
            include_cropped=True,    # Include cropped images of each structure
            include_masks=True,      # Include segmentation masks
            include_metadata=True    # Include JSON and CSV metadata
        )
        
        if result["success"]:
            print("✅ Analysis completed successfully!")
            print(f"\n📊 Results Summary:")
            print(f"   • Total files created: {result['total_files_created']}")
            print(f"   • Bones detected: {result['structure_summary']['total_bones']}")
            print(f"   • Organs detected: {result['structure_summary']['total_organs']}")
            print(f"   • Total structures: {result['structure_summary']['total_bones'] + result['structure_summary']['total_organs']}")
            
            print(f"\n🦴 Detected Bones:")
            for bone in result['structure_summary']['bone_list']:
                print(f"   - {bone}")
            
            print(f"\n🫁 Detected Organs:")
            for organ in result['structure_summary']['organ_list']:
                print(f"   - {organ}")
            
            print(f"\n📁 Output Structure:")
            print(f"   {output_folder}/")
            print(f"   ├── visualizations/     ({len(result['created_files']['visualizations'])} files)")
            print(f"   ├── cropped_images/     ({len(result['created_files']['cropped_images'])} files)")
            print(f"   ├── masks/              ({len(result['created_files']['masks'])} files)")
            print(f"   ├── metadata/           ({len(result['created_files']['metadata'])} files)")
            print(f"   └── README.txt")
            
            print(f"\n🎯 Key Files Created:")
            print(f"   • Main visualization: {output_folder}/visualizations/all_structures_bbox.png")
            print(f"   • Bounding box data: {output_folder}/metadata/bounding_boxes_data.json")
            print(f"   • Summary CSV: {output_folder}/metadata/bounding_boxes_summary.csv")
            print(f"   • Analysis guide: {output_folder}/README.txt")
            
        else:
            print(f"❌ Analysis failed: {result['error']}")
            return
        
        # Additional test: Get specific structure bounding box
        print(f"\n🔍 Testing specific structure detection...")
        
        # Test getting heart bounding box specifically
        heart_result = anatomy_tool.get_structure_bounding_box(
            image_path=input_image,
            structure_name="heart"
        )
        
        if heart_result["success"]:
            bbox = heart_result["bounding_box"]["bounding_box"]
            print(f"❤️  Heart detected at:")
            print(f"   Position: ({bbox['min_x']}, {bbox['min_y']}) to ({bbox['max_x']}, {bbox['max_y']})")
            print(f"   Size: {bbox['width']} x {bbox['height']} pixels")
            print(f"   Center: ({bbox['center_x']}, {bbox['center_y']})")
            print(f"   Area: {heart_result['bounding_box']['area_pixels']} pixels")
        else:
            print(f"❌ Heart detection failed: {heart_result['error']}")
            if "available_structures" in heart_result:
                print(f"Available structures: {heart_result['available_structures']}")
        
        # Test CTR calculation
        print(f"\n📏 Testing Cardio-Thoracic Ratio (CTR) calculation...")
        ctr_result = anatomy_tool.calculate_ctr(input_image)
        
        if ctr_result["success"] and ctr_result["ctr_value"] is not None:
            print(f"📊 CTR Analysis:")
            print(f"   • CTR Value: {ctr_result['ctr_value']:.3f}")
            print(f"   • Interpretation: {ctr_result['interpretation']}")
        else:
            print(f"❌ CTR calculation failed or unavailable")
            if ctr_result.get("error"):
                print(f"   Error: {ctr_result['error']}")
        
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        anatomy_tool.cleanup()
        
        print(f"\n✅ Test completed! Check the '{output_folder}' folder for results.")
        print(f"📋 Open '{output_folder}/README.txt' for detailed information about the analysis.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_anatomy_segmentation()