# XrayAgent 🔬

An intelligent X-ray analysis agent that automatically selects and uses appropriate medical imaging tools based on natural language queries and input images. Powered by Azure OpenAI GPT-4.1.

## Features ✨

- **Intelligent Tool Selection**: Automatically chooses the best tool(s) for your query
- **Natural Language Interface**: Ask questions in plain English
- **Multiple Analysis Types**: Supports classification, segmentation, detection, VQA, and measurements
- **Azure OpenAI Integration**: Uses GPT-4.1 for intelligent reasoning and analysis
- **Extensible Architecture**: Easy to add new tools and capabilities

## Available Tools 🔧

The agent currently supports these specialized X-ray analysis tools:

| Tool | Description | Capabilities |
|------|-------------|--------------|
| **MedGemma-VQA** | Medical Visual Question Answering | Answer complex medical questions about X-rays |
| **TorchXrayVision** | Chest X-ray pathology classification | Detect 14+ pathologies including pneumonia, cardiomegaly |
| **FactCheXcker CarinaNet** | ETT positioning detection | Detect endotracheal tube and carina positioning |
| **ChestXRayAnatomySegmentation** | Anatomical structure segmentation | Segment 157 anatomical structures, calculate CTR |
| **MAIRA-2** | Grounding and localization | Locate and ground medical findings |

## Quick Start 🚀

### 1. Installation

#### Option A: Automatic Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd XrayAgent

# Run the setup script
python setup.py
```

#### Option B: Manual Setup
```bash
# Clone the repository
git clone <repository-url>
cd XrayAgent

# Install basic dependencies
pip install -r requirements.txt

# Test the installation
python test_basic.py
```

### 2. Configuration

#### Option A: Using the Environment Template (Recommended)
```bash
# Use the provided template
cp env.example .env

# Edit .env with your actual Azure OpenAI credentials
# Required: AZURE_OPENAI_API_KEY=your_actual_api_key_here
# Optional: AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME, etc.
```

#### Option B: Manual Environment Setup
Create a `.env` file in the project root with your Azure OpenAI credentials:

```bash
# Create .env file
cat > .env << EOF
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_DEPLOYMENT_NAME=your-deployment-name
EOF
```

Or set environment variables directly:
```bash
export AZURE_OPENAI_API_KEY="your_actual_api_key_here"
```

**Note**: The agent will use default values for endpoint and deployment if not specified.

### 3. Usage

#### Command Line Interface

```bash
# Basic usage
python demo.py --image path/to/xray.jpg --query "What pathologies do you see?"

# List available tools
python demo.py --list-tools

# Verbose output
python demo.py --image xray.jpg --query "Is there pneumonia?" --verbose
```

#### Interactive Mode

```bash
# Start interactive mode (no arguments)
python demo.py
```

#### Python API

```python
from src.xray_agent import XrayAgent

# Initialize the agent
agent = XrayAgent()

# Process a query
results = agent.process_query("path/to/xray.jpg", "What pathologies do you see?")

# Print results
print(results["summary"])
```

## Example Queries 💬

The agent can handle various types of medical queries:

### Diagnostic Questions
- "What pathologies do you see in this chest X-ray?"
- "Is there evidence of pneumonia?"
- "Do you see any signs of cardiac enlargement?"
- "Are there any lung nodules or masses?"

### Measurement Questions  
- "What is the cardio-thoracic ratio?"
- "Can you measure the heart size?"
- "What are the anatomical measurements?"

### Positioning Questions
- "Is the endotracheal tube positioned correctly?"
- "How far is the ETT tip from the carina?"
- "Is the central line placement appropriate?"

### Educational Questions
- "What are the key findings in this X-ray?"
- "Can you explain the abnormalities you see?"
- "What would be your differential diagnosis?"

### Structural Analysis
- "Can you segment the anatomical structures?"
- "What organs are visible in this image?"
- "Identify the heart, lungs, and other structures"

## How It Works 🧠

1. **Query Analysis**: GPT-4.1 analyzes your question and the X-ray image
2. **Tool Selection**: The AI automatically selects the most appropriate tool(s)
3. **Execution**: Selected tools are executed in the optimal order
4. **Integration**: Results are combined and interpreted
5. **Summary**: A comprehensive summary is generated

## Architecture 🏗️

```
XrayAgent/
├── src/
│   ├── xray_agent.py          # Main agent class
│   └── tools/                 # Tool definitions
│       ├── Xray_VQA_MedGemma.json
│       ├── Xray_Class_TorchXrayVision.json
│       ├── Xray_Detection_FactCheXckerCarinaNet.json
│       ├── Xray_Seg_ChestXRayAnatomySegmentation.json
│       └── ...
├── demo.py                    # Demo script
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Output Format 📊

The agent returns structured results including:

```json
{
  "query": "What pathologies do you see?",
  "image_path": "path/to/xray.jpg",
  "analysis": {
    "selected_tools": ["MedGemma-VQA", "TorchXrayVision"],
    "reasoning": "The query asks for pathology detection...",
    "query_type": "diagnostic",
    "confidence": 0.95
  },
  "tool_results": {
    "MedGemma-VQA": {
      "answer": "The chest X-ray shows...",
      "confidence_level": "High"
    },
    "TorchXrayVision": {
      "predictions": {
        "pathology_scores": {
          "Pneumonia": 0.12,
          "Cardiomegaly": 0.34,
          "Atelectasis": 0.08
        }
      }
    }
  },
  "summary": "Comprehensive analysis summary..."
}
```

## Customization 🛠️

### Adding New Tools

1. Create a JSON file in `src/tools/` with the tool specification
2. Follow the existing format with `input_schema`, `output_schema`, etc.
3. Add execution logic in `XrayAgent.execute_tool()` method

### Modifying Tool Selection

The tool selection logic is in `XrayAgent.analyze_query()`. You can:
- Modify the system prompt to change selection criteria
- Add new capability keywords in `_get_tool_capabilities()`
- Adjust the confidence thresholds

## Testing 🧪

### Basic Tests (No API Key Required)
```bash
# Test imports and basic functionality
python test_basic.py
```

### Full Tests (Requires API Key)
```bash
# Test with actual Azure OpenAI integration
python test_agent.py
```

## Troubleshooting 🔧

### Import Errors
If you see import errors like `Import "PIL" could not be resolved`:

```bash
# Install missing packages
pip install Pillow>=9.0.0 openai>=1.0.0

# Or reinstall all requirements
pip install -r requirements.txt
```

### API Key Issues
If you see `Please set AZURE_OPENAI_API_KEY environment variable`:

1. Create a `.env` file with your API key:
   ```
   AZURE_OPENAI_API_KEY=your_actual_key_here
   ```
2. Or set the environment variable:
   ```bash
   export AZURE_OPENAI_API_KEY="your_key_here"
   ```

### Path Issues
If you see `ModuleNotFoundError: No module named 'xray_agent'`:

1. Make sure you're running scripts from the project root directory
2. The `src/` directory should be automatically added to the Python path

### Package Installation Issues
For ML packages that fail to install:

```bash
# Install without ML dependencies (basic functionality only)
pip install openai Pillow numpy pandas requests python-dotenv

# For full functionality, install PyTorch first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-image
```

## Current Limitations ⚠️

- **Simulated Results**: Tool executions are currently simulated for demonstration
- **Tool Dependencies**: Actual tool execution requires specific model downloads and GPU setup
- **Image Formats**: Currently supports JPEG, PNG; DICOM support may vary by tool

## Future Enhancements 🔮

- [ ] Real tool execution with actual model inference
- [ ] Support for DICOM images with metadata
- [ ] Batch processing capabilities
- [ ] Web interface for easy access
- [ ] Integration with hospital PACS systems
- [ ] Multi-modal analysis (CT, MRI support)

## Development 👨‍💻

### Running Tests

```bash
# Test tool loading
python -c "from src.xray_agent import XrayAgent; agent = XrayAgent(); print('Tools loaded:', len(agent.tools))"

# Test with sample image
python demo.py --image sample.jpg --query "Test query" --verbose
```

### Adding Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## Contributing 🤝

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Support 🆘

For questions or issues:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. For urgent medical questions, consult a healthcare professional

## Acknowledgments 🙏

- Azure OpenAI team for GPT-4.1 API
- Medical imaging tool authors and researchers
- Open source medical AI community

---

**Disclaimer**: This tool is for research and educational purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions. 