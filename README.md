# xRAYaGENT - Enhanced X-ray Analysis Agent

**Enhanced intelligent X-ray analysis agent powered by GPT-4.1 with dynamic code generation capabilities.**

## 🌟 Enhanced Features

### Dynamic Code Generation
- **GPT-4.1 Code Generation**: Automatically generates Python code based on natural language queries
- **Tool Selection**: Intelligent selection of appropriate medical imaging tools for each query
- **Real-time Execution**: Executes generated code safely with comprehensive error handling
- **Fallback Mechanisms**: Robust fallback strategies when primary tools fail

### Advanced Analysis Pipeline
1. **Query Analysis**: GPT-4.1 analyzes the question and X-ray image
2. **Tool Selection**: Automatically selects optimal tools from available library
3. **Code Generation**: Generates executable Python code using selected tools
4. **Safe Execution**: Runs code in controlled environment with safety controls
5. **Result Processing**: Processes and formats results for user consumption

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/xRAYaGENT.git
cd xRAYaGENT

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Basic Usage
```python
from src.xray_agent import EnhancedXrayAgent

# Initialize the agent
agent = EnhancedXrayAgent()

# Process a query
result = agent.process_query(
    image_path="data/xray.jpg",
    query="What pathologies do you see in this chest X-ray?"
)

# Access results
print(result['analysis']['reasoning'])  # GPT-4.1 analysis
print(result['generated_code'])         # Generated Python code
print(result['result'])                 # Execution results
```

### Demo Script
```bash
# Run the enhanced demo
python demo.py

# This will demonstrate:
# - Dynamic code generation
# - Multiple query types
# - Tool selection logic
# - Real-time execution
# - Error handling
```

## 🛠️ Available Tools

The enhanced agent automatically selects from these medical imaging tools:

### 1. MedGemma-VQA
- **Purpose**: Visual Question Answering for medical images
- **Capabilities**: Natural language responses to medical queries
- **Use Cases**: General pathology detection, clinical interpretation

### 2. TorchXrayVision
- **Purpose**: Deep learning-based pathology classification
- **Capabilities**: Multi-pathology detection with confidence scores
- **Use Cases**: Automated screening, pathology scoring

### 3. ChestXRayAnatomySegmentation
- **Purpose**: Anatomical structure segmentation
- **Capabilities**: Organ segmentation, clinical measurements
- **Use Cases**: Cardiac ratio calculation, anatomical analysis

### 4. FactCheXcker CarinaNet
- **Purpose**: Endotracheal tube positioning detection
- **Capabilities**: ETT detection, carina localization
- **Use Cases**: Critical care monitoring, tube placement verification

### 5. BoneFractureDetection
- **Purpose**: Bone fracture detection in X-rays
- **Capabilities**: Fracture classification, confidence scoring
- **Use Cases**: Emergency radiology, trauma assessment

### 6. MAIRA-2
- **Purpose**: Advanced medical image grounding
- **Capabilities**: Spatial localization, detailed annotations
- **Use Cases**: Teaching, detailed analysis

## 🎯 Example Queries

The enhanced agent can handle various types of medical queries:

### General Analysis
```python
"What pathologies do you see in this chest X-ray?"
"Analyze this X-ray for any abnormalities"
"What is your overall impression of this image?"
```

### Specific Disease Detection
```python
"Is there evidence of pneumonia?"
"Do you see signs of cardiomegaly?"
"Are there any fractures visible?"
```

### Quantitative Analysis
```python
"What is the cardio-thoracic ratio?"
"Can you measure the cardiac silhouette?"
"Give me confidence scores for pathology detection"
```

### Anatomical Assessment
```python
"Can you segment the anatomical structures?"
"What is the cardiac silhouette appearance?"
"Identify the lung boundaries"
```

### Critical Care
```python
"Is the endotracheal tube positioned correctly?"
"Check for proper tube placement"
"Assess the carina position"
```

## 🔧 Technical Architecture

### EnhancedXrayAgent Class
- **Dynamic Code Generation**: Uses GPT-4.1 to generate tool-specific code
- **Safe Execution**: CodeExecutor class provides sandboxed execution environment
- **Tool Management**: Automatic loading and management of medical imaging tools
- **Result Processing**: Comprehensive result analysis and formatting

### CodeExecutor Class
- **Sandboxed Environment**: Controlled execution environment for generated code
- **Error Handling**: Comprehensive error capture and reporting
- **Security**: Restricted module access and safe execution practices
- **Output Capture**: Captures stdout, stderr, and execution results

### Tool Integration
- **JSON Configuration**: Tools defined in JSON files with schemas and examples
- **Dynamic Loading**: Automatic tool discovery and loading
- **Flexible Execution**: Supports various tool types and interfaces
- **Extensible Design**: Easy to add new tools and capabilities

## 📊 Response Format

The enhanced agent returns comprehensive results:

```json
{
  "query": "User's original question",
  "image_path": "Path to analyzed image",
  "analysis": {
    "reasoning": "GPT-4.1 analysis of query and image",
    "selected_tools": ["List of selected tools"],
    "expected_output": "Description of expected results"
  },
  "generated_code": "Python code generated by GPT-4.1",
  "execution": {
    "success": true,
    "stdout": "Execution output",
    "stderr": "Error messages",
    "error": "Error details if failed"
  },
  "result": {
    "answer": "Processed tool results",
    "confidence": "Confidence metrics",
    "findings": "Key medical findings"
  },
  "summary": "Human-readable summary of analysis"
}
```

## 🚨 Safety Features

### Code Execution Safety
- **Sandboxed Environment**: Isolated execution environment
- **Module Restrictions**: Limited access to system modules
- **Timeout Protection**: Execution time limits
- **Error Isolation**: Comprehensive error handling

### Medical Safety
- **Disclaimer**: Not for clinical diagnosis
- **Validation**: Results should be validated by medical professionals
- **Transparency**: Full code and reasoning provided
- **Auditability**: Complete execution logs maintained

## 🔬 Development

### Adding New Tools
1. Create tool JSON definition in `src/tools/`
2. Include demo commands and schemas
3. Tool will be automatically discovered and integrated

### Customizing Behavior
- Modify system prompts in `EnhancedXrayAgent`
- Adjust safety parameters in `CodeExecutor`
- Extend tool capabilities through JSON configuration

## 📈 Performance

### Optimization Features
- **Parallel Processing**: Multiple tool execution where possible
- **Caching**: Result caching for repeated queries
- **Resource Management**: Efficient memory and GPU utilization
- **Scalability**: Designed for high-throughput scenarios

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Tool development
- Code improvements
- Documentation updates
- Bug reports and feature requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [TorchXrayVision](https://github.com/mlmed/torchxrayvision)
- [Medical Imaging Deep Learning](https://github.com/Project-MONAI/MONAI)

---

**⚠️ Important**: This tool is for research and educational purposes only. Always consult qualified medical professionals for clinical diagnosis and treatment decisions. 