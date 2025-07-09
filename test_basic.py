#!/usr/bin/env python3
"""
Basic test script for XrayAgent - Tests core functionality without API calls

This script tests the basic setup and imports without requiring API keys.
"""

import sys
from pathlib import Path
import tempfile
import os

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    test_packages = [
        ("PIL (Pillow)", "from PIL import Image"),
        ("OpenAI", "from openai import AzureOpenAI"),
        ("NumPy", "import numpy"),
        ("Pandas", "import pandas"),
        ("Requests", "import requests"),
        ("JSON", "import json"),
        ("Base64", "import base64"),
        ("Pathlib", "from pathlib import Path"),
    ]
    
    failed_imports = []
    
    for name, import_statement in test_packages:
        try:
            exec(import_statement)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} packages failed to import")
        return False
    else:
        print(f"\n✅ All {len(test_packages)} packages imported successfully")
        return True

def test_file_structure():
    """Test that required files and directories exist"""
    print("\n🧪 Testing file structure...")
    
    required_paths = [
        ("Main agent", "src/xray_agent.py"),
        ("Tools directory", "src/tools/"),
        ("Demo script", "demo.py"),
        ("Requirements", "requirements.txt"),
        ("Data directory", "data/"),
    ]
    
    missing_files = []
    
    for name, path in required_paths:
        if Path(path).exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (missing)")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} required files/directories missing")
        return False
    else:
        print(f"\n✅ All required files and directories found")
        return True

def test_tool_files():
    """Test that tool JSON files exist and are valid"""
    print("\n🧪 Testing tool configuration files...")
    
    tools_dir = Path("src/tools")
    json_files = list(tools_dir.glob("*.json"))
    
    if not json_files:
        print("   ❌ No tool JSON files found")
        return False
    
    valid_tools = 0
    for json_file in json_files:
        try:
            import json
            with open(json_file, 'r') as f:
                tool_data = json.load(f)
            
            # Check for required fields
            required_fields = ["tool_name", "tool_description"]
            if all(field in tool_data for field in required_fields):
                print(f"   ✅ {json_file.name}")
                valid_tools += 1
            else:
                print(f"   ⚠️  {json_file.name} (missing required fields)")
        
        except Exception as e:
            print(f"   ❌ {json_file.name}: {e}")
    
    if valid_tools > 0:
        print(f"\n✅ Found {valid_tools} valid tool configuration files")
        return True
    else:
        print(f"\n❌ No valid tool configuration files found")
        return False

def test_agent_loading():
    """Test that the agent can be imported and instantiated (without API calls)"""
    print("\n🧪 Testing agent loading...")
    
    try:
        # Mock the API key to avoid the error
        os.environ["AZURE_OPENAI_API_KEY"] = "test_key_for_basic_testing"
        
        from xray_agent import XrayAgent
        print("   ✅ XrayAgent class imported successfully")
        
        # Try to instantiate (this will load tools but not make API calls)
        try:
            agent = XrayAgent()
            print(f"   ✅ Agent instantiated with {len(agent.tools)} tools")
            
            # Test tool listing
            tools = agent.list_available_tools()
            print(f"   ✅ Tool listing works: {len(tools)} tools available")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Agent instantiation failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    finally:
        # Clean up test environment variable
        if "AZURE_OPENAI_API_KEY" in os.environ and os.environ["AZURE_OPENAI_API_KEY"] == "test_key_for_basic_testing":
            del os.environ["AZURE_OPENAI_API_KEY"]

def main():
    """Run basic tests"""
    print("🔬 XrayAgent Basic Test Suite")
    print("=" * 50)
    print("This test checks core functionality without requiring API keys\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure), 
        ("Tool Configuration", test_tool_files),
        ("Agent Loading", test_agent_loading),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All basic tests passed!")
        print("\n🚀 Next steps:")
        print("   1. Set up your Azure OpenAI API key in a .env file:")
        print("      AZURE_OPENAI_API_KEY=your_actual_key_here")
        print("   2. Run the full test: python test_agent.py")
        print("   3. Try the demo: python demo.py --image data/xray.jpg --query 'What do you see?'")
    elif passed_tests > 0:
        print("⚠️  Some tests passed, but there are issues to resolve")
        print("   Check the error messages above for details")
    else:
        print("❌ All tests failed. Please check your installation:")
        print("   1. Ensure you're in the XrayAgent directory")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Or run the setup script: python setup.py")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 