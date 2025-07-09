#!/usr/bin/env python3
"""
Demo script for XrayAgent - Intelligent X-ray Analysis Agent

This script demonstrates how to use the XrayAgent to analyze chest X-ray images
with natural language queries using Azure OpenAI GPT-4.1.

Usage:
    python demo.py --image ./data/xray.jpg --query "What is the cardiac silhouette size?"
    python demo.py --image path/to/xray.jpg --query "Is there pneumonia?"
    python demo.py --image path/to/xray.jpg --query "What is the cardiac silhouette size?"
"""

import argparse
import json
import sys
from pathlib import Path
import logging
import os

# Add src to path to import our agent
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from xray_agent import XrayAgent
except ImportError as e:
    print(f"❌ Error importing XrayAgent: {e}")
    print("Make sure you're running this script from the project root directory")
    print("and that all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="XrayAgent Demo - Intelligent X-ray Analysis")
    parser.add_argument("--image", "-i", required=True, 
                       help="Path to the chest X-ray image file")
    parser.add_argument("--query", "-q", required=True,
                       help="Natural language query about the X-ray")
    parser.add_argument("--tools-dir", default="src/tools",
                       help="Directory containing tool JSON files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--list-tools", action="store_true",
                       help="List available tools and exit")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize the agent
        print("🔬 Initializing XrayAgent...")
        agent = XrayAgent(tools_dir=args.tools_dir)
        
        # List tools if requested
        if args.list_tools:
            print("\n📋 Available Tools:")
            print("=" * 50)
            tools = agent.list_available_tools()
            for name, info in tools.items():
                print(f"\n🔧 {name}")
                print(f"   Description: {info['description'][:100]}...")
                print(f"   Tasks: {', '.join(info['supported_tasks'])}")
                print(f"   Capabilities: {', '.join(info['capabilities'])}")
            return
        
        # Validate image file
        if not Path(args.image).exists():
            print(f"❌ Error: Image file '{args.image}' not found")
            return
        
        print(f"📷 Image: {args.image}")
        print(f"❓ Query: {args.query}")
        print("\n🤖 Processing query with XrayAgent...")
        print("=" * 60)
        
        # Process the query
        results = agent.process_query(args.image, args.query)
        
        # Display results
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Show analysis
        analysis = results.get("analysis", {})
        print(f"\n🧠 AI Analysis:")
        print(f"   Query Type: {analysis.get('query_type', 'Unknown')}")
        print(f"   Confidence: {analysis.get('confidence', 0):.2f}")
        print(f"   Selected Tools: {', '.join(analysis.get('selected_tools', []))}")
        print(f"   Reasoning: {analysis.get('reasoning', 'No reasoning provided')}")
        
        # Show tool results
        print(f"\n🔧 Tool Results:")
        print("=" * 40)
        
        tool_results = results.get("tool_results", {})
        for tool_name, tool_result in tool_results.items():
            print(f"\n📊 {tool_name}:")
            
            if "error" in tool_result:
                print(f"   ❌ Error: {tool_result['error']}")
                continue
            
            if tool_name == "MedGemma-VQA":
                print(f"   💬 Answer: {tool_result.get('answer', 'No answer')}")
                print(f"   🎯 Confidence: {tool_result.get('confidence_level', 'Unknown')}")
                findings = tool_result.get('key_findings', [])
                if findings:
                    print(f"   🔍 Key Findings: {', '.join(findings)}")
            
            elif tool_name == "TorchXrayVision":
                predictions = tool_result.get("predictions", {}).get("pathology_scores", {})
                if predictions:
                    print("   🩺 Pathology Predictions:")
                    # Sort by confidence score
                    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    for pathology, score in sorted_preds[:5]:  # Show top 5
                        print(f"      {pathology}: {score:.3f}")
            
            elif tool_name == "ChestXRayAnatomySegmentation":
                structures = tool_result.get("anatomical_structures", [])
                if structures:
                    print(f"   🫁 Anatomical Structures: {', '.join(structures)}")
                
                measurements = tool_result.get("clinical_measurements", {})
                if measurements:
                    print("   📏 Clinical Measurements:")
                    for measure, value in measurements.items():
                        print(f"      {measure.replace('_', ' ').title()}: {value}")
            
            elif tool_name == "FactCheXcker CarinaNet":
                print(f"   🫁 ETT Detected: {tool_result.get('ett_detected', 'Unknown')}")
                print(f"   🔍 Carina Detected: {tool_result.get('carina_detected', 'Unknown')}")
                print(f"   📍 Positioning: {tool_result.get('positioning_status', 'Unknown')}")
                if 'carina_confidence' in tool_result:
                    print(f"   🎯 Carina Confidence: {tool_result['carina_confidence']:.2f}")
            
            # Show status
            status = tool_result.get("status", "unknown")
            if status == "simulated":
                print("   ⚠️  Note: This is a simulated result for demonstration")
        
        # Show summary
        summary = results.get("summary", "")
        if summary:
            print(f"\n📝 Summary:")
            print("=" * 30)
            print(summary)
        
        # Option to save results
        print(f"\n💾 Full results available in JSON format")
        save_choice = input("Save detailed results to file? (y/n): ").lower().strip()
        if save_choice == 'y':
            output_file = f"xray_analysis_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def interactive_mode():
    """Interactive mode for multiple queries"""
    print("🔬 XrayAgent Interactive Mode")
    print("=" * 40)
    
    # Get image path
    while True:
        image_path = input("📷 Enter path to X-ray image: ").strip()
        if Path(image_path).exists():
            break
        print("❌ File not found. Please try again.")
    
    # Initialize agent
    print("\n🤖 Initializing XrayAgent...")
    agent = XrayAgent()
    
    print("\n✅ Agent ready! Enter your queries (type 'quit' to exit)")
    print("Example queries:")
    print("  - What pathologies do you see in this chest X-ray?")
    print("  - Is there evidence of pneumonia?")
    print("  - What is the cardio-thoracic ratio?")
    print("  - Can you segment the anatomical structures?")
    print()
    
    while True:
        try:
            query = input("❓ Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\n🤖 Processing: {query}")
            print("-" * 50)
            
            results = agent.process_query(image_path, query)
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
                continue
            
            # Quick summary
            analysis = results.get("analysis", {})
            tools_used = ', '.join(analysis.get('selected_tools', []))
            print(f"🔧 Tools used: {tools_used}")
            
            # Show key results
            tool_results = results.get("tool_results", {})
            for tool_name, tool_result in tool_results.items():
                if tool_name == "MedGemma-VQA":
                    answer = tool_result.get('answer', 'No answer')
                    print(f"💬 {answer}")
                elif tool_name == "TorchXrayVision":
                    predictions = tool_result.get("predictions", {}).get("pathology_scores", {})
                    if predictions:
                        top_pred = max(predictions.items(), key=lambda x: x[1])
                        print(f"🩺 Top finding: {top_pred[0]} ({top_pred[1]:.3f})")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, start interactive mode
        interactive_mode()
    else:
        main() 