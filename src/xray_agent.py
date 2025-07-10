import os
import json
import glob
import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Try to import required packages with helpful error messages
try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL (Pillow) is required. Install with: pip install Pillow>=9.0.0")

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError("OpenAI library is required. Install with: pip install openai>=1.0.0")

# Import tool classes
from tools.torchxrayvision_classifier import TorchXrayVisionClassifier
from tools.anatomy_segmentation import ChestXrayAnatomySegmentation
from tools.ett_detection import ETTDetection
from tools.bone_fracture_detection import BoneFractureDetection
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
            "BoneFractureDetection": BoneFractureDetection(),
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
            "BoneFractureDetection": {
                "detect_fractures": {
                    "description": "Detect bone fractures in X-ray image",
                    "parameters": ["image_path", "confidence_threshold"]
                },
                "assess_fracture_severity": {
                    "description": "Assess the severity of detected fractures",
                    "parameters": ["image_path"]
                },
                "analyze_bone_health": {
                    "description": "Analyze overall bone health and density",
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

    def select_functions(self, query: str, image_path: str) -> Dict[str, Any]:
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
    - question: Use the user's query or modified version for VQA
    - context: Add relevant clinical context if needed
    - threshold: Use 0.5 as default confidence threshold
    - model_type: Use "densenet121-res224-all" as default for TorchXrayVision
    - return_masks: Use false unless specifically requested
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
            
            # Try to extract JSON from response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                analysis = json.loads(json_str)
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

    def process_query(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Main method to process a query with an image using predefined functions
        """
        logger.info(f"Processing query: '{query}' for image: {image_path}")
        
        # Validate inputs
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Select functions using GPT-4.1
        function_selection = self.select_functions(query, image_path)
        
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
        
        # Prepare comprehensive response
        response = {
            "query": query,
            "image_path": image_path,
            "analysis": analysis,
            "results": structured_answer,
            "summary": structured_answer.get("answer", "No answer generated")
        }
        
        return response

    def list_available_functions(self) -> Dict[str, Any]:
        """Return information about all available functions"""
        return self.available_functions


# Example usage
if __name__ == "__main__":
    # Initialize the  agent
    agent = XrayAgent()

    result = agent.process_query("../data/xray.jpg", "Is there evidence of ETT in this X-ray? If so, where is it located?")
    print(json.dumps(result, indent=2, default=str)) 