#!/usr/bin/env python3
"""
XRAYaGENT Demo Script

This script demonstrates the capabilities of the XRAYaGENT system for medical imaging analysis.
It processes a sample chest X-ray using multiple AI tools and generates comprehensive reports.

Features:
- Multi-tool AI analysis
- Real model inference
- Comprehensive reporting
- Tool testing capabilities

Usage:
    python demo.py
"""

import os
import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from xray_agent import XrayAgent

def main():
    """Main demo function"""
    print("🔬 Enhanced XrayAgent Demo - Predefined Functions")
    print("=" * 60)
    
    # Initialize the enhanced agent
    print("Initializing XrayAgent...")
    try:
        agent = XrayAgent()
        print(f"✅ Successfully initialized agent with predefined functions")
        print()
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Show available functions
    print("🛠️  Available Functions:")
    functions = agent.list_available_functions()
    for tool_name, tool_functions in functions.items():
        print(f"  📦 {tool_name}:")
        for func_name, func_info in tool_functions.items():
            print(f"    • {func_name}: {func_info['description']}")
    print()
    
    # Check for sample image
    image_path = "data/xray.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Sample image not found: {image_path}")
        print("Please ensure the X-ray image exists in the data directory.")
        return
    
    # Sample queries to demonstrate different capabilities
    sample_queries = [
        {
            "query": "What pathologies do you see in this chest X-ray?",
            "description": "VQA for general pathology detection"
        },
        {
            "query": "Can you classify the pathologies with confidence scores?",
            "description": "Pathology classification with scoring"
        },
        {
            "query": "What is the cardio-thoracic ratio?",
            "description": "Anatomical measurements"
        },
        {
            "query": "Is the endotracheal tube positioned correctly?",
            "description": "ETT positioning assessment"
        },
        {
            "query": "Are there any bone fractures visible?",
            "description": "Fracture detection"
        }
    ]
    
    print("🔍 Running Demo Queries...")
    print("=" * 60)
    
    for i, query_info in enumerate(sample_queries, 1):
        query = query_info["query"]
        description = query_info["description"]
        
        print(f"\n📋 Query {i}: {description}")
        print(f"❓ Question: {query}")
        print("-" * 40)
        
        try:
            # Run the query
            result = agent.process_query(image_path, query)
            
            # Display results
            print(f"\nQuery: {result['query']}")
            print(f"Image: {result['image_path']}")
            print(f"\nAnalysis:")
            print(f"  Reasoning: {result['analysis']['reasoning']}")
            print(f"  Selected Functions: {', '.join(result['analysis']['selected_functions'])}")
            
            print(f"\nResults:")
            print(f"  Answer: {result['results']['answer']}")
            print(f"  Key Findings: {', '.join(result['results']['key_findings'])}")
            print(f"  Confidence: {result['results']['confidence']}")
            if result['results']['recommendations']:
                print(f"  Recommendations: {', '.join(result['results']['recommendations'])}")
            if result['results']['technical_notes']:
                print(f"  Technical Notes: {result['results']['technical_notes']}")
            
            # Print summary
            print(f"\nSummary: {result['summary']}")
            
            # Save results to file
            with open('xray_analysis_results.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print("\n" + "="*60)
            print("Analysis complete. Results saved to 'xray_analysis_results.json'")
            print("="*60)
            
            # Demo: Folder processing for anatomy segmentation
            print("\n🔍 Demonstrating folder processing for anatomy segmentation...")
            try:
                # Use the agent to process a folder
                folder_query = "Process all images in the data folder and generate anatomy masks"
                folder_result = agent.process_query("data/", folder_query)
                
                print(f"\nFolder Query: {folder_result['query']}")
                print(f"Analysis: {folder_result['analysis']['reasoning']}")
                print(f"Results: {folder_result['results']['answer']}")
                
                # Save folder results
                with open('folder_analysis_results.json', 'w') as f:
                    json.dump(folder_result, f, indent=2, default=str)
                
                print("\n" + "="*60)
                print("Folder processing complete. Results saved to 'folder_analysis_results.json'")
                print("="*60)
                
            except Exception as e:
                print(f"⚠️  Folder processing demo failed: {e}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("-" * 60)
    
    print("\n🎯 Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• GPT-4.1 selection of appropriate predefined functions")
    print("• Structured parameter passing to medical imaging tools")
    print("• Real tool execution with proper error handling")
    print("• Comprehensive result analysis and reporting")
    print("• Modular, testable tool architecture")

def test_individual_tools():
    """Test individual tool functions"""
    print("\n🧪 Testing Individual Tool Functions")
    print("=" * 50)
    
    # Test each tool individually
    tool_tests = [
        ("TorchXrayVision", "torchxrayvision_classifier", "test_torchxrayvision"),
        ("Anatomy Segmentation", "anatomy_segmentation", "test_anatomy_segmentation"),
        ("ETT Detection", "ett_detection", "test_ett_detection"),
        ("Bone Fracture Detection", "bone_fracture_detection", "test_bone_fracture_detection"),
        ("MAIRA-2 Detection", "maira_2", "test_maira2_detection"),
    ]
    
    sys.path.append(str(Path(__file__).parent / "src" / "tools"))
    
    for tool_name, module_name, test_func_name in tool_tests:
        print(f"\n🔧 Testing {tool_name}...")
        try:
            module = __import__(module_name)
            test_func = getattr(module, test_func_name)
            success = test_func()
            if success:
                print(f"✅ {tool_name} test passed")
            else:
                print(f"❌ {tool_name} test failed")
        except Exception as e:
            print(f"❌ {tool_name} test error: {e}")

def show_function_selection_example():
    """Show how GPT-4.1 selects functions for a query"""
    print("\n🤖 GPT-4.1 Function Selection Example")
    print("=" * 50)
    
    agent = XrayAgent()
    image_path = "data/xray.jpg"
    
    if not os.path.exists(image_path):
        print("❌ Sample image not found")
        return
    
    query = "What pathologies do you see and what is the cardio-thoracic ratio?"
    
    print(f"Query: {query}")
    print("\nFunction Selection Process:")
    print("-" * 30)
    
    try:
        function_selection = agent.select_functions(query, image_path)
        
        print(f"Reasoning: {function_selection.get('reasoning', 'No reasoning provided')}")
        print(f"Expected Output: {function_selection.get('expected_output', 'No description')}")
        print("\nSelected Function Calls:")
        
        for i, func_call in enumerate(function_selection.get("function_calls", []), 1):
            print(f"  {i}. {func_call['tool_name']}.{func_call['function_name']}")
            print(f"     Parameters: {func_call['parameters']}")
        
    except Exception as e:
        print(f"Error in function selection: {e}")

if __name__ == "__main__":
    # Run main demo
    main()
    
    # Optional individual tool testing
    test_tools = input("\n🔬 Would you like to test individual tools? (y/n): ")
    if test_tools.lower() == 'y':
        test_individual_tools()
    
    # Optional function selection example
    show_selection = input("\n🤖 Would you like to see function selection example? (y/n): ")
    if show_selection.lower() == 'y':
        show_function_selection_example() 