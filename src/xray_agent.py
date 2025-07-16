import os
import json
import glob
import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import re

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
    from openai import AzureOpenAI
except ImportError:
    raise ImportError("OpenAI library is required. Install with: pip install openai>=1.0.0")

# Import tool classes
from .tools.torchxrayvision_classifier import TorchXrayVisionClassifier
from .tools.anatomy_segmentation import ChestXrayAnatomySegmentation
from .tools.ett_detection import ETTDetection
from .tools.maira_2 import MAIRA2Detection

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
            "MAIRA-2": MAIRA2Detection()
        }
    
    def execute_function(self, function_call: FunctionCall) -> Dict[str, Any]:
        """Execute a function call"""
        try:
            tool = self.tools.get(function_call.tool_name)
            if not tool:
                return {"error": f"Tool {function_call.tool_name} not found"}
            
            # Get the function from the tool
            function = getattr(tool, function_call.function_name, None)
            if not function:
                return {"error": f"Function {function_call.function_name} not found in {function_call.tool_name}"}
            
            # Call the function with parameters
            result = function(**function_call.parameters)
            return result
            
        except Exception as e:
            logger.error(f"Error executing function: {e}")
            return {"error": str(e)}

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
                    "description": "Segment anatomical structures with structured output directory (../output/study_id/question_id/imasks)",
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
            output_dir = Path(f"../output/{study_id}/{question_id}")
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
    - This saves masks to organized directories: ../output/study_id/question_id/imasks
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
        
        system_prompt = """You are an expert radiologist providing clear, concise answers to medical questions about X-ray images. Your task is to synthesize the results from multiple medical imaging tools into a single, coherent answer.

Guidelines:
1. Answer the user's original question directly and clearly
2. Use the tool results to support your answer
3. Provide a structured response with key findings
4. Use medical terminology appropriately but ensure clarity
5. If results are conflicting, acknowledge the discrepancy
6. Only include information relevant to answering the question

Response Format:
Return a JSON object with:
- "answer": Direct answer to the user's question (2-3 sentences)
- "key_findings": List of main findings relevant to the question
- "confidence": Overall confidence level (High/Moderate/Low)
- "recommendations": Any relevant clinical recommendations (if applicable)
- "technical_notes": Brief technical details if relevant to the answer"""

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
                
                # Plot bounding box
                bbox_image_path = self.plot_bounding_box(
                    image_path=image_path,
                    bounding_box=bbox_coords,
                    study_id=study_id,
                    question_id=question_id,
                    query=query,
                    confidence=confidence
                )
        
        # Prepare comprehensive response
        response = {
            "query": query,
            "image_path": image_path,
            "study_id": study_id,
            "question_id": question_id,
            "analysis": analysis,
            "results": structured_answer,
            "summary": structured_answer.get("answer", "No answer generated"),
            "bbox_image_path": bbox_image_path
        }
        
        return response

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
    print("Check ../output/ directory for:")
    print("- Bounding box images: ../output/study_id/question_id/img_with_bbox.png")
    print("- Anatomy masks: ../output/study_id/question_id/imasks/")
    print("=" * 60) 