import os
import json
import glob
import subprocess
import tempfile
import base64
from typing import Dict, List, Any, Optional, Tuple
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
class ToolInfo:
    """Dataclass to store tool information"""
    name: str
    description: str
    author: str
    version: str
    input_schema: dict
    output_schema: dict
    demo_commands: List[str]
    file_path: str
    supported_tasks: List[str]

class XrayAgent:
    """
    Intelligent X-ray analysis agent that automatically selects and uses appropriate tools
    based on input images and natural language queries.
    """
    
    def __init__(self, tools_dir: str = "src/tools"):
        """
        Initialize the XrayAgent
        
        Args:
            tools_dir: Directory containing tool JSON files
        """
        self.tools_dir = Path(tools_dir)
        self.client = get_azure_client()
        self.tools = {}
        self.load_tools()
        
    def load_tools(self):
        """Load all available tools from JSON files"""
        logger.info(f"Loading tools from {self.tools_dir}")
        
        tool_files = glob.glob(str(self.tools_dir / "*.json"))
        
        for tool_file in tool_files:
            try:
                with open(tool_file, 'r') as f:
                    tool_data = json.load(f)
                
                # Extract supported tasks from tool description and name
                supported_tasks = self._extract_supported_tasks(tool_data)
                
                tool_info = ToolInfo(
                    name=tool_data.get("tool_name", ""),
                    description=tool_data.get("tool_description", ""),
                    author=tool_data.get("tool_author", ""),
                    version=tool_data.get("tool_version", ""),
                    input_schema=tool_data.get("input_schema", {}),
                    output_schema=tool_data.get("output_schema", {}),
                    demo_commands=tool_data.get("demo_commands", []),
                    file_path=tool_file,
                    supported_tasks=supported_tasks
                )
                
                self.tools[tool_info.name] = tool_info
                logger.info(f"Loaded tool: {tool_info.name}")
                
            except Exception as e:
                logger.error(f"Error loading tool from {tool_file}: {e}")
        
        logger.info(f"Successfully loaded {len(self.tools)} tools")

    def _extract_supported_tasks(self, tool_data: dict) -> List[str]:
        """Extract supported tasks from tool data"""
        tasks = []
        name = tool_data.get("tool_name", "").lower()
        description = tool_data.get("tool_description", "").lower()
        
        # Map common keywords to task types
        task_keywords = {
            "classification": ["class", "classify", "pathology", "disease"],
            "detection": ["detect", "localization", "bbox", "object"],
            "segmentation": ["segment", "mask", "anatomy", "structure"],
            "vqa": ["question", "answering", "vqa", "query"],
            "measurement": ["measure", "distance", "ratio", "metric"],
            "grounding": ["grounding", "localize", "position"]
        }
        
        for task, keywords in task_keywords.items():
            if any(keyword in name or keyword in description for keyword in keywords):
                tasks.append(task)
        
        return tasks if tasks else ["general"]

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API calls"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_query(self, query: str, image_path: str) -> Dict[str, Any]:
        """
        Use GPT-4.1 to analyze the query and determine which tools to use
        
        Args:
            query: User's natural language query
            image_path: Path to the input image
            
        Returns:
            Dictionary containing tool selection and reasoning
        """
        # Prepare tool descriptions for GPT
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_desc = {
                "name": name,
                "description": tool.description,
                "supported_tasks": tool.supported_tasks,
                "capabilities": self._get_tool_capabilities(tool)
            }
            tool_descriptions.append(tool_desc)
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        system_prompt = f"""You are an intelligent medical imaging agent that helps select appropriate tools for X-ray analysis based on user queries.

Available tools:
{json.dumps(tool_descriptions, indent=2)}

Your task is to:
1. Analyze the user's query and the provided X-ray image
2. Determine which tool(s) would be most appropriate to answer the query
3. Provide reasoning for your selection
4. Suggest the order of tool execution if multiple tools are needed

Respond with a JSON object containing:
- "selected_tools": List of tool names to use
- "reasoning": Explanation of why these tools were selected
- "execution_order": Suggested order of execution
- "query_type": Classification of the query (e.g., "diagnostic", "measurement", "educational")
- "confidence": Confidence score (0-1) in tool selection
"""

        user_prompt = f"""
Please analyze this chest X-ray image and determine which tools would be best suited to answer the following query:

Query: "{query}"

Consider:
- What specific information is the user seeking?
- Which tools have the capabilities to provide that information?
- Are multiple tools needed to fully answer the query?
- What would be the logical order of execution?
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
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                analysis = json.loads(json_str)
            except:
                # Fallback parsing
                analysis = {
                    "selected_tools": ["MedGemma-VQA"],  # Default to VQA
                    "reasoning": "Fallback to VQA tool due to parsing error",
                    "execution_order": ["MedGemma-VQA"],
                    "query_type": "general",
                    "confidence": 0.5
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Return default analysis
            return {
                "selected_tools": ["MedGemma-VQA"],
                "reasoning": f"Error in analysis, defaulting to VQA: {e}",
                "execution_order": ["MedGemma-VQA"],
                "query_type": "general",
                "confidence": 0.3
            }

    def _get_tool_capabilities(self, tool: ToolInfo) -> List[str]:
        """Extract key capabilities from tool information"""
        capabilities = []
        
        # Extract from description
        description_lower = tool.description.lower()
        
        capability_keywords = {
            "pathology_detection": ["pathology", "disease", "abnormal"],
            "anatomy_identification": ["anatomy", "structure", "organ"],
            "measurement": ["measure", "distance", "ratio", "size"],
            "positioning": ["position", "placement", "location"],
            "question_answering": ["question", "answer", "query", "vqa"],
            "classification": ["classify", "classification", "category"],
            "localization": ["localize", "detect", "find", "locate"],
            "segmentation": ["segment", "mask", "boundary"]
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities if capabilities else ["general_analysis"]

    def execute_tool(self, tool_name: str, image_path: str, query: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute a specific tool with the given parameters
        
        Args:
            tool_name: Name of the tool to execute
            image_path: Path to the input image
            query: Optional query for VQA tools
            **kwargs: Additional parameters for the tool
            
        Returns:
            Dictionary containing tool execution results
        """
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        tool = self.tools[tool_name]
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix=f"{tool_name}_")
            
            if tool_name == "MedGemma-VQA":
                return self._execute_vqa_tool(tool, image_path, query, output_dir)
            elif tool_name == "TorchXrayVision":
                return self._execute_classification_tool(tool, image_path, output_dir)
            elif tool_name == "FactCheXcker CarinaNet":
                return self._execute_detection_tool(tool, image_path, output_dir)
            elif tool_name == "ChestXRayAnatomySegmentation":
                return self._execute_segmentation_tool(tool, image_path, output_dir)
            elif tool_name == "BoneFractureDetection":
                return self._execute_bone_fracture_detection_tool(tool, image_path, output_dir)
            else:
                return self._execute_generic_tool(tool, image_path, output_dir, query)
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Tool execution failed: {e}"}

    def _execute_vqa_tool(self, tool: ToolInfo, image_path: str, query: str, output_dir: str) -> Dict[str, Any]:
        """Execute VQA tool (MedGemma)"""
        # For now, we'll return a simulated response since the actual tool execution
        # requires specific environment setup
        return {
            "tool_name": tool.name,
            "query": query,
            "image_path": image_path,
            "answer": f"[Simulated VQA Response for: {query}] This would contain the actual medical VQA response analyzing the chest X-ray image for the specific question asked.",
            "confidence_level": "Moderate",
            "key_findings": ["Chest X-ray analysis", "Medical interpretation"],
            "status": "simulated"
        }

    def _execute_classification_tool(self, tool: ToolInfo, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Execute classification tool (TorchXrayVision)"""
        return {
            "tool_name": tool.name,
            "image_path": image_path,
            "predictions": {
                "pathology_scores": {
                    "Pneumonia": 0.12,
                    "Cardiomegaly": 0.34,
                    "Atelectasis": 0.08,
                    "Infiltration": 0.15,
                    "Mass": 0.03,
                    "Nodule": 0.07,
                    "Pneumothorax": 0.02
                }
            },
            "status": "simulated"
        }

    def _execute_detection_tool(self, tool: ToolInfo, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Execute detection tool (FactCheXcker CarinaNet)"""
        return {
            "tool_name": tool.name,
            "image_path": image_path,
            "ett_detected": False,
            "carina_detected": True,
            "positioning_status": "No ETT detected",
            "carina_confidence": 0.87,
            "status": "simulated"
        }

    def _execute_segmentation_tool(self, tool: ToolInfo, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Execute segmentation tool (ChestXRayAnatomySegmentation)"""
        return {
            "tool_name": tool.name,
            "image_path": image_path,
            "anatomical_structures": ["Heart", "Left Lung", "Right Lung", "Spine", "Ribs"],
            "clinical_measurements": {
                "cardio_thoracic_ratio": 0.45,
                "spine_center_distance": 124.5
            },
            "status": "simulated"
        }

    def _execute_generic_tool(self, tool: ToolInfo, image_path: str, output_dir: str, query: str) -> Dict[str, Any]:
        """Execute a generic tool"""
        return {
            "tool_name": tool.name,
            "image_path": image_path,
            "query": query,
            "result": f"Generic execution result for {tool.name}",
            "status": "simulated"
        }

    def _execute_bone_fracture_detection_tool(self, tool: ToolInfo, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Execute bone fracture detection tool (BoneFractureDetection)"""
        return {
            "tool_name": tool.name,
            "image_path": image_path,
            "predictions": {
                "fracture_status": "Fractured",
                "confidence": 0.95
            },
            "status": "simulated"
        }

    def process_query(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Main method to process a query with an image
        
        Args:
            image_path: Path to the X-ray image
            query: Natural language query about the image
            
        Returns:
            Comprehensive results from selected tools
        """
        logger.info(f"Processing query: '{query}' for image: {image_path}")
        
        # Validate inputs
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Analyze query to select appropriate tools
        analysis = self.analyze_query(query, image_path)
        
        # Execute selected tools
        results = {
            "query": query,
            "image_path": image_path,
            "analysis": analysis,
            "tool_results": {},
            "summary": ""
        }
        
        for tool_name in analysis.get("execution_order", []):
            logger.info(f"Executing tool: {tool_name}")
            tool_result = self.execute_tool(tool_name, image_path, query)
            results["tool_results"][tool_name] = tool_result
        
        # Generate comprehensive summary
        results["summary"] = self._generate_summary(results)
        
        return results

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of all tool results"""
        summary_parts = []
        
        query = results.get("query", "")
        analysis = results.get("analysis", {})
        
        summary_parts.append(f"Query Analysis: {analysis.get('reasoning', 'No reasoning provided')}")
        summary_parts.append(f"Selected Tools: {', '.join(analysis.get('selected_tools', []))}")
        
        # Summarize each tool's results
        for tool_name, tool_result in results.get("tool_results", {}).items():
            if "error" not in tool_result:
                summary_parts.append(f"\n{tool_name} Results:")
                
                if tool_name == "MedGemma-VQA":
                    summary_parts.append(f"  Answer: {tool_result.get('answer', 'No answer')}")
                elif tool_name == "TorchXrayVision":
                    scores = tool_result.get("predictions", {}).get("pathology_scores", {})
                    top_findings = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    summary_parts.append(f"  Top pathology predictions: {top_findings}")
                elif tool_name == "ChestXRayAnatomySegmentation":
                    ctr = tool_result.get("clinical_measurements", {}).get("cardio_thoracic_ratio", "N/A")
                    summary_parts.append(f"  Cardio-Thoracic Ratio: {ctr}")
                elif tool_name == "FactCheXcker CarinaNet":
                    status = tool_result.get("positioning_status", "Unknown")
                    summary_parts.append(f"  ETT Positioning: {status}")
        
        return "\n".join(summary_parts)

    def list_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Return information about all available tools"""
        tool_info = {}
        for name, tool in self.tools.items():
            tool_info[name] = {
                "description": tool.description,
                "supported_tasks": tool.supported_tasks,
                "capabilities": self._get_tool_capabilities(tool),
                "author": tool.author,
                "version": tool.version
            }
        return tool_info

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = XrayAgent()
    
    # Example queries
    sample_queries = [
        "What pathologies do you see in this chest X-ray?",
        "Is there evidence of pneumonia?",
        "What is the cardio-thoracic ratio?",
        "Is the endotracheal tube positioned correctly?",
        "Can you segment the anatomical structures?",
        "What is your overall impression of this X-ray?"
    ]
    
    # Print available tools
    print("Available Tools:")
    print(json.dumps(agent.list_available_tools(), indent=2)) 