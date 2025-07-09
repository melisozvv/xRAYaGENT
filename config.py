"""
Configuration file for XrayAgent

This file contains all the configuration settings for the XrayAgent.
Modify these values according to your setup and requirements.
"""

import os
from pathlib import Path

# Azure OpenAI Configuration
# =========================
AZURE_OPENAI_ENDPOINT = "https://azure-ai.hms.edu"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT_NAME = "gpt-4.1"
AZURE_OPENAI_API_KEY = "b960432f5dd540969d3083910b085a33"

# Alternative: Load from environment variables
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")

# Tool Configuration
# ==================
TOOLS_DIRECTORY = "src/tools"  # Directory containing tool JSON files
DEFAULT_OUTPUT_DIR = "output"  # Default directory for tool outputs

# Agent Behavior Settings
# =======================
DEFAULT_TEMPERATURE = 0.1  # Lower = more focused, Higher = more creative
MAX_TOKENS = 1000  # Maximum tokens for GPT responses
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for tool selection

# Supported Image Formats
# =======================
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
MAX_IMAGE_SIZE_MB = 10  # Maximum image file size in MB

# Logging Configuration
# ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model Preferences
# =================
PREFERRED_VQA_MODEL = "MedGemma-VQA"
PREFERRED_CLASSIFICATION_MODEL = "TorchXrayVision"  
PREFERRED_SEGMENTATION_MODEL = "ChestXRayAnatomySegmentation"
PREFERRED_DETECTION_MODEL = "FactCheXcker CarinaNet"

# Tool Selection Keywords
# ======================
TOOL_SELECTION_KEYWORDS = {
    "vqa": ["question", "what", "describe", "explain", "tell me", "analysis"],
    "classification": ["pathology", "disease", "condition", "abnormal", "detect"],
    "segmentation": ["segment", "structure", "anatomy", "outline", "boundary"],
    "detection": ["detect", "find", "locate", "position", "placement"],
    "measurement": ["measure", "size", "ratio", "distance", "calculate"]
}

# Output Settings
# ===============
SAVE_INTERMEDIATE_RESULTS = True  # Save individual tool results
INCLUDE_CONFIDENCE_SCORES = True  # Include confidence in outputs
GENERATE_SUMMARY = True  # Generate comprehensive summary
VERBOSE_OUTPUT = False  # Include detailed processing information

# Advanced Settings
# =================
ENABLE_PARALLEL_EXECUTION = False  # Execute multiple tools in parallel
MAX_CONCURRENT_TOOLS = 2  # Maximum tools to run simultaneously
ENABLE_RESULT_CACHING = False  # Cache results for repeated queries
CACHE_DURATION_HOURS = 24  # How long to cache results

# Safety and Validation
# ====================
VALIDATE_IMAGE_FORMAT = True  # Validate image before processing
REQUIRE_MEDICAL_DISCLAIMER = True  # Show medical disclaimer
MAX_QUERY_LENGTH = 500  # Maximum characters in user query

# Development and Testing
# ======================
ENABLE_SIMULATION_MODE = True  # Use simulated results for demo
DEBUG_MODE = False  # Enable debug logging and features
TEST_MODE = False  # Use test configurations

def get_config():
    """Return configuration as dictionary"""
    return {
        # Azure OpenAI
        "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
        "azure_openai_api_version": AZURE_OPENAI_API_VERSION,
        "azure_deployment_name": AZURE_DEPLOYMENT_NAME,
        "azure_openai_api_key": AZURE_OPENAI_API_KEY,
        
        # Directories
        "tools_directory": TOOLS_DIRECTORY,
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        
        # Agent settings
        "default_temperature": DEFAULT_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        
        # File handling
        "supported_extensions": SUPPORTED_IMAGE_EXTENSIONS,
        "max_image_size_mb": MAX_IMAGE_SIZE_MB,
        
        # Logging
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        
        # Model preferences
        "preferred_models": {
            "vqa": PREFERRED_VQA_MODEL,
            "classification": PREFERRED_CLASSIFICATION_MODEL,
            "segmentation": PREFERRED_SEGMENTATION_MODEL,
            "detection": PREFERRED_DETECTION_MODEL
        },
        
        # Tool selection
        "tool_keywords": TOOL_SELECTION_KEYWORDS,
        
        # Output
        "save_intermediate": SAVE_INTERMEDIATE_RESULTS,
        "include_confidence": INCLUDE_CONFIDENCE_SCORES,
        "generate_summary": GENERATE_SUMMARY,
        "verbose_output": VERBOSE_OUTPUT,
        
        # Advanced
        "parallel_execution": ENABLE_PARALLEL_EXECUTION,
        "max_concurrent": MAX_CONCURRENT_TOOLS,
        "enable_caching": ENABLE_RESULT_CACHING,
        "cache_duration": CACHE_DURATION_HOURS,
        
        # Safety
        "validate_images": VALIDATE_IMAGE_FORMAT,
        "medical_disclaimer": REQUIRE_MEDICAL_DISCLAIMER,
        "max_query_length": MAX_QUERY_LENGTH,
        
        # Development
        "simulation_mode": ENABLE_SIMULATION_MODE,
        "debug_mode": DEBUG_MODE,
        "test_mode": TEST_MODE
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required settings
    if not AZURE_OPENAI_API_KEY:
        errors.append("AZURE_OPENAI_API_KEY is required")
    
    if not AZURE_OPENAI_ENDPOINT:
        errors.append("AZURE_OPENAI_ENDPOINT is required")
    
    # Check directories exist
    if not Path(TOOLS_DIRECTORY).exists():
        errors.append(f"Tools directory not found: {TOOLS_DIRECTORY}")
    
    # Check value ranges
    if not 0 <= DEFAULT_TEMPERATURE <= 2:
        errors.append("DEFAULT_TEMPERATURE must be between 0 and 2")
    
    if not 0 < DEFAULT_CONFIDENCE_THRESHOLD <= 1:
        errors.append("DEFAULT_CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    if MAX_TOKENS < 100:
        errors.append("MAX_TOKENS should be at least 100")
    
    return errors

if __name__ == "__main__":
    # Validate configuration when run directly
    errors = validate_config()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration is valid")
        
    # Print current configuration
    print("\n📋 Current Configuration:")
    config = get_config()
    for key, value in config.items():
        print(f"   {key}: {value}") 