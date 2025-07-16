#!/usr/bin/env python3
"""
Demo script for X-ray Agent with bounding box plotting and structured anatomy segmentation
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the path to import XrayAgent
sys.path.insert(0, str(Path(__file__).parent / "src"))

from xray_agent import XrayAgent

def main():
    """Demonstrate the enhanced X-ray Agent capabilities"""
    
    print("=" * 80)
    print("Enhanced X-ray Agent Demo")
    print("Featuring: Bounding Box Plotting & Structured Anatomy Segmentation")
    print("=" * 80)
    
    # Initialize the agent
    try:
        agent = XrayAgent()
        print("✅ X-ray Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize X-ray Agent: {e}")
        return
    
    # Test image path
    image_path = "data/xray.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        print("Please ensure you have a test X-ray image at data/xray.jpg")
        return
    
    print(f"📸 Using test image: {image_path}")
    print()
    
    # Test cases with different functionalities
    test_cases = [
        {
            "name": "Heart Location Detection",
            "query": "Where is the heart located in this X-ray? Please provide the bounding box coordinates.",
            "study_id": "demo_study_001",
            "question_id": "q1_heart_location",
            "expected_features": ["bounding_box", "location_detection"]
        },
        {
            "name": "Anatomical Structure Segmentation",  
            "query": "Can you segment all the anatomical structures in this chest X-ray?",
            "study_id": "demo_study_002",
            "question_id": "q2_anatomy_segmentation", 
            "expected_features": ["structured_masks", "anatomy_segmentation"]
        },
        {
            "name": "Lung Disease Detection",
            "query": "Is there any evidence of pneumonia or other lung disease? If found, where is it located?",
            "study_id": "demo_study_003",
            "question_id": "q3_disease_detection",
            "expected_features": ["disease_detection", "location_analysis"]
        },
        {
            "name": "ETT Position Analysis",
            "query": "Is there an endotracheal tube present? If so, where is it positioned?",
            "study_id": "demo_study_004", 
            "question_id": "q4_ett_analysis",
            "expected_features": ["ett_detection", "position_analysis"]
        }
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test Case {i}: {test_case['name']}")
        print(f"   Query: {test_case['query']}")
        print(f"   Study ID: {test_case['study_id']}")
        print(f"   Question ID: {test_case['question_id']}")
        
        try:
            # Process the query
            result = agent.process_query(
                image_path=image_path,
                query=test_case['query'],
                study_id=test_case['study_id'],
                question_id=test_case['question_id']
            )
            
            # Store result
            results[test_case['name']] = result
            
            # Display summary
            print(f"   ✅ Processing completed")
            print(f"   📝 Summary: {result.get('summary', 'No summary available')[:100]}...")
            
            # Check for bounding box output
            if result.get('bbox_image_path'):
                print(f"   📦 Bounding box image: {result['bbox_image_path']}")
            
            # Check for structured outputs
            selected_functions = result.get('analysis', {}).get('selected_functions', [])
            if 'segment_anatomy_structured' in selected_functions:
                expected_mask_dir = f"../output/{test_case['study_id']}/{test_case['question_id']}/imasks"
                print(f"   🎭 Anatomy masks should be in: {expected_mask_dir}")
                
        except Exception as e:
            print(f"   ❌ Error processing test case: {e}")
            continue
        
        print()
    
    # Summary
    print("=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    successful_tests = len([r for r in results.values() if 'error' not in r])
    print(f"✅ Successful tests: {successful_tests}/{len(test_cases)}")
    
    print("\n📁 Output Directory Structure:")
    print("../output/")
    for test_case in test_cases:
        study_id = test_case['study_id']
        question_id = test_case['question_id']
        print(f"├── {study_id}/")
        print(f"│   └── {question_id}/")
        print(f"│       ├── img_with_bbox.png  (if bounding box detected)")
        print(f"│       └── imasks/            (if anatomy segmentation performed)")
    
    print("\n🔧 New Features Demonstrated:")
    print("1. ✅ Automatic bounding box plotting for location-based queries")
    print("2. ✅ Structured anatomy segmentation with organized output directories") 
    print("3. ✅ Enhanced query processing with study_id and question_id organization")
    print("4. ✅ Intelligent function selection based on query content")
    
    print("\n💡 Usage Tips:")
    print("- Use location-related keywords ('where', 'locate', 'position') for bounding box detection")
    print("- Anatomy segmentation automatically uses structured output format")
    print("- All outputs are organized by study_id and question_id for easy management")
    print("- Bounding boxes are automatically plotted when coordinates are detected")
    
    print("\n" + "=" * 80)
    print("Demo completed! Check the ../output/ directory for generated files.")
    print("=" * 80)

if __name__ == "__main__":
    main() 