#!/usr/bin/env python3
"""
Setup script for XrayAgent

This script helps install dependencies and configure the environment
for the XrayAgent medical imaging analysis tool.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_basic_requirements():
    """Install basic requirements"""
    return run_command(
        "pip install -r requirements.txt", 
        "Installing basic requirements"
    )

def install_full_requirements():
    """Install all requirements including ML packages"""
    basic_packages = [
        "openai>=1.0.0",
        "Pillow>=9.0.0", 
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0"
    ]
    
    ml_packages = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "scikit-image>=0.18.0"
    ]
    
    # Install basic packages first
    for package in basic_packages:
        if not run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            return False
    
    # Try to install ML packages (these might fail on some systems)
    print("\n🤖 Installing ML packages (this may take a while)...")
    for package in ml_packages:
        if not run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {package.split('>=')[0]}. Some features may not work.")
    
    return True

def create_env_file():
    """Create a sample .env file"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating sample .env file...")
    try:
        with open(env_file, 'w') as f:
            f.write("""# Azure OpenAI Configuration
# Replace with your actual API key
AZURE_OPENAI_API_KEY=your_api_key_here

# Optional: Override default endpoints
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
# AZURE_DEPLOYMENT_NAME=your-deployment-name
""")
        print("✅ Created .env file")
        print("⚠️  Remember to add your actual Azure OpenAI API key to the .env file")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_imports():
    """Test that key imports work"""
    print("🧪 Testing imports...")
    
    test_imports = [
        ("PIL", "from PIL import Image"),
        ("openai", "from openai import AzureOpenAI"),
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
    ]
    
    for name, import_statement in test_imports:
        try:
            exec(import_statement)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🔬 XrayAgent Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("src/xray_agent.py").exists():
        print("❌ Please run this script from the XrayAgent directory")
        print("   Expected file structure: src/xray_agent.py")
        sys.exit(1)
    
    print("\n📦 Installing dependencies...")
    
    # Try full installation first
    if not install_full_requirements():
        print("\n⚠️  Full installation failed, trying basic installation...")
        if not install_basic_requirements():
            print("❌ Failed to install basic requirements")
            sys.exit(1)
    
    # Create environment file
    print("\n🔧 Setting up environment...")
    create_env_file()
    
    # Test imports
    print("\n🧪 Testing installation...")
    if test_imports():
        print("\n✅ Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Add your Azure OpenAI API key to the .env file")
        print("   2. Test the installation: python test_agent.py")
        print("   3. Try a demo: python demo.py --image data/xray.jpg --query 'What do you see?'")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("   Some imports failed. You may need to install additional packages manually.")
    
    print("\n📖 For more information, see the README.md file")

if __name__ == "__main__":
    main() 