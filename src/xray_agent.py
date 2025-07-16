import os
import json
import glob
import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import re
from datetime import datetime

# Try to import required packages with helpful error messages
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("PIL (Pillow) is required. Install with: pip install Pillow>=9.0.0")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import colors
    import numpy as np
except ImportError:
    raise ImportError("Matplotlib and NumPy are required. Install with: pip install matplotlib numpy")

try:
    import cv2
except ImportError:
    logger.warning("OpenCV not found. IOU calculation will be disabled. Install with: pip install opencv-python")

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError("OpenAI library is required. Install with: pip install openai>=1.0.0")

# Import tool classes
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from tools.covid19 import COVID19Detection
from tools.torchxrayvision_classifier import TorchXrayVisionClassifier
from tools.anatomy_segmentation import ChestXrayAnatomySegmentation
from tools.ett_detection import ETTDetection
from tools.maira_2 import MAIRA2Detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Using system environment variables only.")

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-ai.hms.edu")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

def get_azure_client():
    """Initialize and return Azure OpenAI client"""
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("Please set AZURE_OPENAI_API_KEY environment variable")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

@dataclass
class FunctionCall:
    """Represents a function call with parameters"""
    tool_name: str
    function_name: str
    parameters: Dict[str, Any]

class FunctionExecutor:
    """Executes predefined functions from tool classes"""
    
    def __init__(self):
        self.tools = {
            "TorchXrayVision": TorchXrayVisionClassifier(),
            "ChestXRayAnatomySegmentation": ChestXrayAnatomySegmentation(),
            "FactCheXcker CarinaNet": ETTDetection(),
            "MAIRA-2": MAIRA2Detection(),
            "COVID19Detection": COVID19Detection()
        }
        self.execution_log = []  # 记录每次执行的详细信息
    
    def execute_function(self, function_call: FunctionCall) -> Dict[str, Any]:
        """Execute a function call and log detailed results"""
        execution_start = datetime.now()
        
        # 创建执行记录
        execution_record = {
            "timestamp": execution_start.isoformat(),
            "tool_name": function_call.tool_name,
            "function_name": function_call.function_name,
            "parameters": function_call.parameters.copy(),
            "status": "started"
        }
        
        try:
            tool = self.tools.get(function_call.tool_name)
            if not tool:
                error_result = {"error": f"Tool {function_call.tool_name} not found"}
                execution_record.update({
                    "status": "error",
                    "error": error_result["error"],
                    "execution_time_ms": 0,
                    "result": error_result
                })
                self.execution_log.append(execution_record)
                return error_result
            
            # Get the function from the tool
            function = getattr(tool, function_call.function_name, None)
            if not function:
                error_result = {"error": f"Function {function_call.function_name} not found in {function_call.tool_name}"}
                execution_record.update({
                    "status": "error", 
                    "error": error_result["error"],
                    "execution_time_ms": 0,
                    "result": error_result
                })
                self.execution_log.append(execution_record)
                return error_result
            
            # 记录执行开始
            logger.info(f"🚀 执行模型: {function_call.tool_name}.{function_call.function_name}")
            logger.info(f"📝 参数: {function_call.parameters}")
            
            # Call the function with parameters
            result = function(**function_call.parameters)
            
            # 计算执行时间
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds() * 1000
            
            # 记录成功执行
            execution_record.update({
                "status": "completed",
                "execution_time_ms": round(execution_time, 2),
                "result": result,
                "completed_at": execution_end.isoformat()
            })
            
            # 记录结果摘要
            result_summary = self._create_result_summary(result)
            execution_record["result_summary"] = result_summary
            
            logger.info(f"✅ Model Run Completed: {function_call.tool_name}.{function_call.function_name}")
            logger.info(f"⏱️ Time: {execution_time:.2f}ms")
            logger.info(f"📊 Summary: {result_summary}")
            
            # 保存详细结果到日志
            self.execution_log.append(execution_record)
            
            return result
            
        except Exception as e:
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds() * 1000
            
            error_msg = str(e)
            execution_record.update({
                "status": "error",
                "error": error_msg,
                "execution_time_ms": round(execution_time, 2),
                "result": {"error": error_msg},
                "completed_at": execution_end.isoformat()
            })
            
            logger.error(f"❌ 模型执行失败: {function_call.tool_name}.{function_call.function_name}")
            logger.error(f"💥 错误: {error_msg}")
            
            self.execution_log.append(execution_record)
            return {"error": error_msg}
    
    def _create_result_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """创建结果摘要用于记录"""
        summary = {}
        
        if "error" in result:
            summary["has_error"] = True
            summary["error"] = result["error"]
            return summary
        
        summary["has_error"] = False
        
        # 统计结果中的关键信息
        if "predicted_class" in result:
            summary["predicted_class"] = result["predicted_class"]
        if "confidence" in result:
            summary["confidence"] = result["confidence"]
        if "covid_probability" in result:
            summary["covid_probability"] = result["covid_probability"]
        if "risk_level" in result:
            summary["risk_level"] = result["risk_level"]
        if "grounding_result" in result:
            summary["has_grounding"] = True
        if "coordinates" in result:
            summary["has_coordinates"] = True
        if "segmentation_completed" in result:
            summary["segmentation_completed"] = result["segmentation_completed"]
        
        # 统计输出文件数量
        if "output_masks" in result:
            summary["mask_count"] = len(result["output_masks"])
        
        return summary
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """获取执行日志"""
        return self.execution_log.copy()
    
    def clear_execution_log(self):
        """清空执行日志"""
        self.execution_log.clear()
    
    def save_execution_log(self, filepath: str):
        """保存执行日志到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.execution_log, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"📁 执行日志已保存到: {filepath}")
        except Exception as e:
            logger.error(f"❌ 保存执行日志失败: {e}")

class XrayAgent:
    """
     X-ray analysis agent that uses GPT-4.1 to select predefined functions
    from medical imaging tools based on natural language queries and images.
    """
    
    def __init__(self, tools_dir: str = "./tools"):
        """Initialize the  XrayAgent"""
        self.tools_dir = Path(tools_dir)
        self.client = get_azure_client()
        self.function_executor = FunctionExecutor()
        self.available_functions = self._get_available_functions()
        
    def _get_available_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available functions"""
        functions = {
            "TorchXrayVision": {
                "classify_pathologies": {
                    "description": "Classify pathologies in chest X-ray",
                    "parameters": ["image_path", "model_type", "threshold"]
                },
                "compare_models": {
                    "description": "Compare different model predictions on the same image",
                    "parameters": ["image_path", "model_types"]
                },
                "get_model_info": {
                    "description": "Get information about a specific model",
                    "parameters": ["model_type"]
                }
            },
            "ChestXRayAnatomySegmentation": {
                "segment_anatomy": {
                    "description": "Segment anatomical structures in chest X-ray",
                    "parameters": ["image_path", "return_masks"]
                },
                "segment_anatomy_structured": {
                    "description": "Segment anatomical structures with structured output directory (/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output/study_id/q{question_id}/imasks)",
                    "parameters": ["image_path", "study_id", "question_id", "return_masks"]
                },
                "process_folder": {
                    "description": "Process all images in a folder and generate mask images",
                    "parameters": ["folder_path", "return_masks"]
                },
                "calculate_clinical_measurements": {
                    "description": "Calculate clinical measurements from segmentation",
                    "parameters": ["image_path"]
                },
                "analyze_structure_positions": {
                    "description": "Analyze anatomical structure positions and relationships",
                    "parameters": ["image_path"]
                }
            },
            "FactCheXcker CarinaNet": {
                "detect_ett_and_carina": {
                    "description": "Detect endotracheal tube and carina in chest X-ray",
                    "parameters": ["image_path", "confidence_threshold"]
                },
                "assess_ett_positioning": {
                    "description": "Assess ETT positioning quality",
                    "parameters": ["image_path"]
                }
            },
            "MAIRA-2": {
                "ground_phrase": {
                    "description": "Ground a medical phrase in chest X-ray image",
                    "parameters": ["image_path", "phrase"]
                },
                "detect_multiple_phrases": {
                    "description": "Ground multiple medical phrases in chest X-ray",
                    "parameters": ["image_path", "phrases"]
                },
                "detect_common_findings": {
                    "description": "Detect common chest X-ray findings using predefined phrases",
                    "parameters": ["image_path"]
                },
                "ground_anatomical_structures": {
                    "description": "Ground anatomical structures in chest X-ray",
                    "parameters": ["image_path"]
                }
            },
            "COVID19Detection": {
                "detect_covid19": {
                    "description": "Detect COVID-19 in chest X-ray image using BEiT vision transformer",
                    "parameters": ["image_path", "return_probabilities"]
                },
                "analyze_covid_risk": {
                    "description": "Analyze COVID-19 risk with detailed interpretation and risk levels",
                    "parameters": ["image_path"]
                },
                "batch_detect": {
                    "description": "Detect COVID-19 in multiple chest X-ray images",
                    "parameters": ["image_paths", "return_probabilities"]
                },
                "get_model_info": {
                    "description": "Get information about the COVID-19 detection model",
                    "parameters": []
                }
            }
        }
        return functions

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API calls"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def plot_bounding_box(self, image_path: str, bounding_box: List[float], study_id: str, question_id: str, 
                         query: str = "", confidence: float = None) -> str:
        """
        Plot bounding box on the original image and save to output directory
        
        Args:
            image_path: Path to the original X-ray image
            bounding_box: List of [x_topleft, y_topleft, x_bottomright, y_bottomright]
            study_id: Study ID for organizing output
            question_id: Question ID for organizing output
            query: Question text for annotation
            confidence: Confidence score to display
            
        Returns:
            Path to the saved image with bounding box
        """
        try:
            # Create output directory
            output_dir = Path(f"/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output/{study_id}/q{question_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Parse bounding box coordinates
            x1, y1, x2, y2 = bounding_box
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # Create matplotlib figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image, cmap='gray')
            
            # Create bounding box rectangle
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add text annotation
            annotation_text = ""
            if query:
                # Truncate long queries
                truncated_query = query[:50] + "..." if len(query) > 50 else query
                annotation_text += f"Q: {truncated_query}\n"
            
            if confidence is not None:
                annotation_text += f"Confidence: {confidence:.3f}"
            
            if annotation_text:
                ax.text(x1, y1 - 10, annotation_text, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                       fontsize=10, verticalalignment='top')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Study: {study_id} | Question: {question_id}", fontsize=12)
            
            # Save image
            output_path = output_dir / "img_with_bbox.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            logger.info(f"Bounding box image saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting bounding box: {e}")
            return ""

    def check_requires_bounding_box(self, query: str) -> bool:
        """
        Check if the question requires bounding box output
        
        Args:
            query: The question text
            
        Returns:
            Boolean indicating if bounding box is required
        """
        query_lower = query.lower()
        bbox_keywords = [
            "bounding_box", "bounding box", "bbox", "locate", "where", 
            "position", "coordinates", "find", "spot", "region"
        ]
        return any(keyword in query_lower for keyword in bbox_keywords)

    def extract_bounding_box_from_result(self, result: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract bounding box coordinates from analysis result
        
        Args:
            result: The analysis result dictionary
            
        Returns:
            List of bounding box coordinates or None if not found
        """
        # Check various possible keys for bounding box
        possible_keys = ["BOUNDING_BOX", "bounding_box", "bbox", "coordinates"]
        
        for key in possible_keys:
            if key in result:
                bbox = result[key]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        return [float(coord) for coord in bbox]
                    except (ValueError, TypeError):
                        continue
        
        return None

    def _post_process_function_calls(self, function_calls: List[Dict[str, Any]], 
                                   study_id: str, question_id: str) -> List[Dict[str, Any]]:
        """
        Post-process function calls to ensure structured methods have required parameters
        
        Args:
            function_calls: List of function call dictionaries
            study_id: Study ID to inject
            question_id: Question ID to inject
            
        Returns:
            Updated function calls with injected parameters
        """
        structured_methods = ["segment_anatomy_structured"]
        
        for func_call in function_calls:
            if func_call.get("function_name") in structured_methods:
                # Ensure parameters dict exists
                if "parameters" not in func_call:
                    func_call["parameters"] = {}
                
                # Inject study_id and question_id if missing
                if "study_id" not in func_call["parameters"]:
                    func_call["parameters"]["study_id"] = study_id
                
                if "question_id" not in func_call["parameters"]:
                    func_call["parameters"]["question_id"] = question_id
        
        return function_calls

    def select_functions(self, query: str, image_path: str, study_id: str = "default", question_id: str = "q1") -> Dict[str, Any]:
        """
        Use GPT-4.1 to select appropriate functions and parameters for the query
        """
        # Encode image for analysis
        base64_image = self.encode_image(image_path)
        
        system_prompt = f"""You are an expert medical imaging AI assistant. Your task is to analyze a medical query and X-ray image, then select the most appropriate predefined functions to answer the question.

    Available Functions:
    {json.dumps(self.available_functions, indent=2)}

    Your responsibilities:
    1. Analyze the user's query and the X-ray image
    2. Select function(s) needed to answer the question
    3. Determine the correct parameters for each function call
    4. Return a structured response with function calls

    Response Format:
    Return a JSON object with:
    - "reasoning": Explanation of function selection and approach
    - "function_calls": List of function calls with tool_name, function_name, and parameters
    - "expected_output": Description of what the functions should produce

    Parameter Guidelines:
    - image_path: Always use "{image_path}"
    - study_id: Use "{study_id}" for structured outputs
    - question_id: Use "{question_id}" for structured outputs
    - question: Use the user's query or modified version for VQA
    - context: Add relevant clinical context if needed
    - threshold: Use 0.5 as default confidence threshold
    - model_type: Use "densenet121-res224-all" as default for TorchXrayVision
    - return_masks: Use false unless specifically requested
    
    Structured Output Guidelines:
    - For anatomy segmentation, prefer "segment_anatomy_structured" over "segment_anatomy" 
    - This saves masks to organized directories: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output/study_id/q{question_id}/imasks
    - Always include study_id and question_id parameters for structured methods
    """

        user_prompt = f"""
Analyze this chest X-ray image and select the appropriate functions to answer the following query:

Query: "{query}"

Requirements:
1. Select the most appropriate function(s) from the available options
2. Determine the correct parameters for each function call
3. Consider what type of analysis is needed (VQA, classification, segmentation, etc.)
4. Return the function calls in a logical order

The response should be in JSON format with the structure specified above.
"""

        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Post-process function calls to ensure structured methods have required parameters
                if "function_calls" in analysis:
                    analysis["function_calls"] = self._post_process_function_calls(
                        analysis["function_calls"], study_id, question_id
                    )
                
            except:
                # Fallback to VQA
                analysis = {
                    "reasoning": "JSON parsing failed, defaulting to VQA",
                    "function_calls": [{
                        "tool_name": "TorchXrayVision",
                        "function_name": "classify_pathologies",
                        "parameters": {
                            "image_path": image_path,
                            "model_type": "densenet121-res224-all",
                            "threshold": 0.5
                        }
                    }],
                    "expected_output": "VQA response"
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in function selection: {e}")
            return {
                "reasoning": f"Error occurred: {e}",
                "function_calls": [{
                    "tool_name": "TorchXrayVision",
                    "function_name": "classify_pathologies",
                    "parameters": {
                        "image_path": image_path,
                        "model_type": "densenet121-res224-all",
                        "threshold": 0.5
                    }
                }],
                "expected_output": "Fallback VQA response"
            }


    def _calculate_bbox_mask_iou(self, bbox: List[float], mask_path: str, original_image_size: tuple = None) -> float:
        """
        Calculate Intersection over Union (IOU) between bounding box and anatomy mask
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            mask_path: Path to the mask image file
            original_image_size: (width, height) of original image if bbox needs scaling
            
        Returns:
            IOU score between 0 and 1
        """
        try:
            import cv2
            
            if not os.path.exists(mask_path):
                logger.warning(f"Mask file not found: {mask_path}")
                return 0.0
            
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not load mask: {mask_path}")
                return 0.0
            
            h, w = mask.shape
            
            # Create bounding box mask
            x1, y1, x2, y2 = bbox
            
            # Scale coordinates to mask size if original image size is provided
            if original_image_size:
                orig_w, orig_h = original_image_size
                x1 = int(x1 * w / orig_w)
                y1 = int(y1 * h / orig_h)
                x2 = int(x2 * w / orig_w)
                y2 = int(y2 * h / orig_h)
            else:
                # Assume coordinates are already in mask coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Create bbox mask
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            bbox_mask[y1:y2, x1:x2] = 255
            
            # Binarize anatomy mask
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Calculate intersection and union
            intersection = cv2.bitwise_and(bbox_mask, binary_mask)
            union = cv2.bitwise_or(bbox_mask, binary_mask)
            
            intersection_area = np.sum(intersection > 0)
            union_area = np.sum(union > 0)
            
            if union_area == 0:
                return 0.0
            
            iou = intersection_area / union_area
            return float(iou)
            
        except Exception as e:
            logger.error(f"Error calculating IOU: {e}")
            return 0.0

    def _determine_location_from_masks(self, bbox: List[float], study_id: str, question_id: str, 
                                     original_image_size: tuple = None) -> Optional[str]:
        """
        Determine anatomical location by calculating IOU with anatomy masks
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            study_id: Study ID for finding mask directory
            question_id: Question ID for finding mask directory
            original_image_size: (width, height) of original image
            
        Returns:
            Location string or None if no good match found
        """
        try:
            # Define mask file mappings
            mask_mappings = {
                "left lung.png": "left lung",
                "right lung.png": "right lung", 
                "lung upper lobe left.png": "lung upper lobe left",
                "lung upper lobe right.png": "lung upper lobe right",
                "lung lower lobe left.png": "lung lower lobe left",
                "lung lower lobe right.png": "lung lower lobe right"
            }
            
            base_output_dir = "/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output"
            mask_dir = f"{base_output_dir}/{study_id}/q{question_id}/imasks"
            
            if not os.path.exists(mask_dir):
                logger.warning(f"Mask directory not found: {mask_dir}")
                return None
            
            best_iou = 0.0
            best_location = None
            
            # Check each possible mask file
            for mask_file, location in mask_mappings.items():
                mask_path = os.path.join(mask_dir, mask_file)
                
                if os.path.exists(mask_path):
                    iou = self._calculate_bbox_mask_iou(bbox, mask_path, original_image_size)
                    logger.info(f"IOU for {location}: {iou:.3f}")
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_location = location
            
            # Only return location if IOU is above threshold
            if best_iou > 0.1:  # 10% overlap threshold
                logger.info(f"Best location match: {best_location} (IOU: {best_iou:.3f})")
                return best_location
            else:
                logger.info(f"No good location match found (best IOU: {best_iou:.3f})")
                return None
                
        except Exception as e:
            logger.error(f"Error determining location from masks: {e}")
            return None

    def synthesize_results(self, query: str, image_path: str, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use GPT-4.1 to synthesize function results into a structured answer
        """
        # Encode image for final analysis
        base64_image = self.encode_image(image_path)
        
        # Prepare results summary for GPT
        results_summary = []
        for result_data in results:
            func_call = result_data["function_call"]
            result = result_data["result"]
            
            if "error" not in result:
                results_summary.append({
                    "function": f"{func_call['tool_name']}.{func_call['function_name']}",
                    "result": result
                })
            else:
                results_summary.append({
                    "function": f"{func_call['tool_name']}.{func_call['function_name']}",
                    "error": result["error"]
                })
   
        

            # Use the specific JSON format required by the question
            system_prompt = f"""You are an expert radiologist providing clear, concise answers to medical questions about X-ray images. Your task is to synthesize the results from multiple medical imaging tools into a single, coherent answer.

CRITICAL INSTRUCTION: The user has specified a required JSON response format. You MUST follow this exact format:

Guidelines:
1. Answer the user's original question directly using the EXACT JSON format specified
2. Use the tool results to support your answer
3. Fill in the required fields with appropriate medical findings
4. Use medical terminology appropriately but ensure clarity
5. If results are conflicting, choose the most reliable findings
6. Only include information relevant to answering the question
7. STRICTLY follow the specified JSON structure - do not add extra fields or change the format

IMPORTANT: Your response must be ONLY the JSON object in the specified format, nothing else."""
       

        user_prompt = f"""
        Original Question: "{query}"

        Analysis Reasoning: {analysis.get('reasoning', 'No reasoning provided')}

        Tool Results:
        {json.dumps(results_summary, indent=2, default=str)}

        Please provide a structured answer to the original question based on these tool results and your analysis of the X-ray image.
        """

        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                structured_answer = json.loads(json_str)
            except:
                # Fallback structure
                structured_answer = {
                    "answer": response_text,
                    "key_findings": ["Analysis completed"],
                    "confidence": "Moderate",
                    "recommendations": [],
                    "technical_notes": "JSON parsing failed, returning raw response"
                }
            
            return structured_answer
            
        except Exception as e:
            logger.error(f"Error in result synthesis: {e}")
            return {
                "answer": f"Unable to synthesize results due to error: {e}",
                "key_findings": ["Error in synthesis"],
                "confidence": "Low",
                "recommendations": [],
                "technical_notes": str(e)
            }

    def process_query(self, image_path: str, query: str, study_id: str = "default", question_id: str = "q1") -> Dict[str, Any]:
        """
        Main method to process a query with an image using predefined functions
        """
        logger.info(f"Processing query: '{query}' for image: {image_path}")
        
        # Validate inputs
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Select functions using GPT-4.1
        function_selection = self.select_functions(query, image_path, study_id, question_id)
        
        # Execute selected functions
        function_results = []
        for func_call_data in function_selection.get("function_calls", []):
            func_call = FunctionCall(
                tool_name=func_call_data["tool_name"],
                function_name=func_call_data["function_name"],
                parameters=func_call_data["parameters"]
            )
            
            result = self.function_executor.execute_function(func_call)
            function_results.append({
                "function_call": func_call_data,
                "result": result
            })
        
        # Synthesize results using GPT-4.1
        analysis = {
            "reasoning": function_selection.get("reasoning", ""),
            "selected_functions": [fc["function_name"] for fc in function_selection.get("function_calls", [])],
            "expected_output": function_selection.get("expected_output", "")
        }
        
        structured_answer = self.synthesize_results(query, image_path, analysis, function_results)
        
        # Check if segment_anatomy_structured was used
        has_anatomy_segmentation = any(
            fc["function_name"] == "segment_anatomy_structured" 
            for fc in function_selection.get("function_calls", [])
        )
        
        # Check if bounding box plotting is needed
        bbox_image_path = ""
        if self.check_requires_bounding_box(query):
            # Look for bounding box in the synthesized answer
            bbox_coords = self.extract_bounding_box_from_result(structured_answer)
            
            if bbox_coords:
                # Get confidence if available
                confidence = structured_answer.get("confidence", None)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence.lower().replace("high", "0.9").replace("moderate", "0.7").replace("low", "0.5"))
                    except:
                        confidence = None
                
                # If anatomy segmentation was used and we have bounding box, determine location via IOU
                if has_anatomy_segmentation:
                    # Get original image size for coordinate scaling
                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            original_image_size = img.size  # (width, height)
                    except:
                        original_image_size = None
                    
                    # Determine location from masks
                    location = self._determine_location_from_masks(
                        bbox_coords, study_id, question_id, original_image_size
                    )
                    
                    if location:
                        # Add LOCATION to structured answer if required JSON format exists
                        if "LOCATION" in structured_answer.upper():
                            structured_answer["LOCATION"] = location
                            logger.info(f"Added LOCATION to response: {location}")
                
                # Plot bounding box
                bbox_image_path = self.plot_bounding_box(
                    image_path=image_path,
                    bounding_box=bbox_coords,
                    study_id=study_id,
                    question_id=question_id,
                    query=query,
                    confidence=confidence
                )
        
        # 获取详细执行记录
        execution_log = self.function_executor.get_execution_log()
        
        # 保存模型执行日志到文件
        if execution_log:
            log_filename = f"model_execution_log_{study_id}_{question_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            base_output_dir = "/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output"
            log_filepath = f"{base_output_dir}/{study_id}/q{question_id}/{log_filename}"
            
            # 确保目录存在
            os.makedirs(f"{base_output_dir}/{study_id}/q{question_id}", exist_ok=True)
            
            # 保存详细日志
            self.function_executor.save_execution_log(log_filepath)
        
        # Prepare comprehensive response
        response = {
            "query": query,
            "image_path": image_path,
            "study_id": study_id,
            "question_id": question_id,
            "analysis": analysis,
            "results": structured_answer,
            "summary": structured_answer.get("answer", "No answer generated"),
            "bbox_image_path": bbox_image_path,
            "model_execution_log": execution_log,  # 包含详细的模型执行记录
            "execution_summary": self._create_execution_summary(execution_log)  # 执行摘要
        }
        
        # 清空执行日志，为下次查询做准备
        self.function_executor.clear_execution_log()
        
        return response
    
    def _create_execution_summary(self, execution_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建执行摘要"""
        if not execution_log:
            return {"total_models": 0, "successful": 0, "failed": 0, "total_time_ms": 0}
        
        total_models = len(execution_log)
        successful = sum(1 for log in execution_log if log.get("status") == "completed")
        failed = sum(1 for log in execution_log if log.get("status") == "error")
        total_time = sum(log.get("execution_time_ms", 0) for log in execution_log)
        
        # 统计使用的模型
        models_used = list(set(f"{log['tool_name']}.{log['function_name']}" for log in execution_log))
        
        return {
            "total_models": total_models,
            "successful": successful,
            "failed": failed,
            "total_time_ms": round(total_time, 2),
            "models_used": models_used,
            "average_time_ms": round(total_time / total_models, 2) if total_models > 0 else 0
        }

    def list_available_functions(self) -> Dict[str, Any]:
        """Return information about all available functions"""
        return self.available_functions


# Example usage
if __name__ == "__main__":
    # Initialize the  agent
    agent = XrayAgent()

    # Example 1: Query with bounding box detection
    print("=" * 60)
    print("Example 1: Query with bounding box detection")
    print("=" * 60)
    result1 = agent.process_query(
        image_path="../data/xray.jpg", 
        query="Where is the heart located in this X-ray?",
        study_id="study_001",
        question_id="q1_heart_location"
    )
    print(f"Query: {result1['query']}")
    print(f"Summary: {result1['summary']}")
    print(f"Bounding box image: {result1['bbox_image_path']}")
    
    # Example 2: Anatomy segmentation with structured output
    print("\n" + "=" * 60)
    print("Example 2: Anatomy segmentation with structured output")
    print("=" * 60)
    result2 = agent.process_query(
        image_path="../data/xray.jpg", 
        query="Can you segment the anatomical structures in this chest X-ray?",
        study_id="study_002", 
        question_id="q2_anatomy_segmentation"
    )
    print(f"Query: {result2['query']}")
    print(f"Summary: {result2['summary']}")
    
    # Example 3: Disease detection
    print("\n" + "=" * 60)
    print("Example 3: Disease detection")
    print("=" * 60)
    result3 = agent.process_query(
        image_path="../data/xray.jpg", 
        query="Is there evidence of pneumonia in this X-ray? If so, where is it located?",
        study_id="study_003",
        question_id="q3_pneumonia_detection"
    )
    print(f"Query: {result3['query']}")
    print(f"Summary: {result3['summary']}")
    print(f"Bounding box image: {result3['bbox_image_path']}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("Check /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output/ directory for:")
    print("- Bounding box images: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output/study_id/q{question_id}/img_with_bbox.png")
    print("- Anatomy masks: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output/study_id/q{question_id}/imasks/")
    print("=" * 60) 