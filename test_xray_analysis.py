#!/usr/bin/env python3
"""
Comprehensive Test Script for Chest X-Ray Anatomy Segmentation and Bounding Box Analysis
This script tests the bounding box functionality with your chest X-ray image.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the XrayAgent module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'XrayAgent', 'src', 'tools'))

def setup_test_environment():
    """Set up the test environment and check requirements"""
    print("🔧 Setting up test environment...")
    
    # Check if the anatomy segmentation module exists
    anatomy_segmentation_path = os.path.join('XrayAgent', 'src', 'tools', 'anatomy_segmentation.py')
    if not os.path.exists(anatomy_segmentation_path):
        print(f"❌ Error: {anatomy_segmentation_path} not found!")
        return False
    
    print("✅ Anatomy segmentation module found")
    return True

def check_image_file():
    """Check if the X-ray image file exists"""
    image_path = "xray.jpg"
    
    if os.path.exists(image_path):
        # Check if it's a real image file (not a text file)
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            print(f"✅ Found valid X-ray image: {image_path} ({width}x{height})")
            return True
        except Exception as e:
            print(f"❌ Invalid image file: {e}")
            return False
    else:
        print(f"❌ X-ray image not found: {image_path}")
        return False

def save_sample_instructions():
    """Create instructions for saving the X-ray image"""
    instructions = """
INSTRUCTIONS FOR SETTING UP THE CHEST X-RAY IMAGE
===============================================

To run the anatomy segmentation test, you need to:

1. Save the chest X-ray image you provided as 'xray.jpg' in this directory:
   /Users/melisozvardar/Desktop/xRAYaGENT/

2. The image should be the chest X-ray you showed (the one with bilateral lung infiltrates).

3. Once saved, run this script again: python test_xray_analysis.py

The test will create a comprehensive analysis folder called 'xray_masks' containing:
- Bounding box visualizations for all bones and organs
- Individual cropped images of each detected structure  
- Segmentation masks
- JSON and CSV metadata files
- A detailed README explaining the results

Expected structures to be detected:
🦴 Bones: ribs, spine, clavicle, sternum, scapula
🫁 Organs: heart, lungs, aorta, mediastinum, diaphragm
"""
    
    with open("XRAY_SETUP_INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)
    
    print("📋 Created setup instructions in XRAY_SETUP_INSTRUCTIONS.txt")

def run_anatomy_analysis():
    """Run the complete anatomy segmentation analysis"""
    try:
        # Import the anatomy segmentation module
        from anatomy_segmentation import ChestXrayAnatomySegmentation
        
        print("\n🔬 Starting Comprehensive Chest X-Ray Analysis")
        print("=" * 50)
        
        # Initialize the tool
        anatomy_tool = ChestXrayAnatomySegmentation()
        
        # Define paths
        input_image = "xray.jpg"
        output_folder = "xray_masks"
        
        print(f"📁 Input: {input_image}")
        print(f"📁 Output: {output_folder}")
        
        # Remove existing output folder if it exists
        if os.path.exists(output_folder):
            print(f"🗑️  Removing existing output folder...")
            shutil.rmtree(output_folder)
        
        print("\n🔍 Performing anatomy segmentation and bounding box analysis...")
        
        # Run the comprehensive analysis
        result = anatomy_tool.create_bounding_boxes_folder(
            image_path=input_image,
            output_folder=output_folder,
            target_structures=None,  # Analyze all structures
            include_cropped=True,    # Include cropped images
            include_masks=True,      # Include segmentation masks  
            include_metadata=True    # Include JSON/CSV metadata
        )
        
        if result["success"]:
            print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("\n📊 RESULTS SUMMARY:")
            print(f"   • Total files created: {result['total_files_created']}")
            print(f"   • Bones detected: {result['structure_summary']['total_bones']}")
            print(f"   • Organs detected: {result['structure_summary']['total_organs']}")
            
            print("\n🦴 DETECTED BONES:")
            for bone in result['structure_summary']['bone_list']:
                print(f"   - {bone}")
            
            print("\n🫁 DETECTED ORGANS:")
            for organ in result['structure_summary']['organ_list']:
                print(f"   - {organ}")
            
            print(f"\n📁 OUTPUT FOLDER STRUCTURE:")
            print(f"   {output_folder}/")
            print(f"   ├── visualizations/     ({len(result['created_files']['visualizations'])} files)")
            print(f"   │   ├── all_structures_bbox.png    ← Main visualization")
            print(f"   │   └── [individual structure images...]")
            print(f"   ├── cropped_images/     ({len(result['created_files']['cropped_images'])} files)")
            print(f"   │   └── [cropped regions of each structure...]")
            print(f"   ├── masks/              ({len(result['created_files']['masks'])} files)")
            print(f"   │   └── [segmentation masks...]")
            print(f"   ├── metadata/           ({len(result['created_files']['metadata'])} files)")
            print(f"   │   ├── bounding_boxes_data.json")
            print(f"   │   └── bounding_boxes_summary.csv")
            print(f"   └── README.txt")
            
            # Test specific structure detection
            print(f"\n🎯 TESTING SPECIFIC STRUCTURE DETECTION:")
            
            # Test heart detection
            heart_result = anatomy_tool.get_structure_bounding_box(input_image, "heart")
            if heart_result["success"]:
                bbox = heart_result["bounding_box"]["bounding_box"]
                print(f"❤️  Heart Location:")
                print(f"   • Position: ({bbox['min_x']}, {bbox['min_y']}) to ({bbox['max_x']}, {bbox['max_y']})")
                print(f"   • Size: {bbox['width']} × {bbox['height']} pixels")
                print(f"   • Center: ({bbox['center_x']}, {bbox['center_y']})")
            
            # Test CTR calculation
            print(f"\n📏 CARDIO-THORACIC RATIO (CTR) ANALYSIS:")
            ctr_result = anatomy_tool.calculate_ctr(input_image)
            if ctr_result["success"] and ctr_result["ctr_value"]:
                print(f"   • CTR Value: {ctr_result['ctr_value']:.3f}")
                print(f"   • Clinical Interpretation: {ctr_result['interpretation']}")
            else:
                print(f"   • CTR calculation not available")
            
            print(f"\n🎉 SUCCESS! Open '{output_folder}/README.txt' for detailed information.")
            print(f"🖼️  View '{output_folder}/visualizations/all_structures_bbox.png' to see all bounding boxes!")
            
        else:
            print(f"❌ Analysis failed: {result['error']}")
            return False
        
        # Cleanup
        anatomy_tool.cleanup()
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("   pip install pillow numpy")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the complete test"""
    print("🩻 CHEST X-RAY ANATOMY SEGMENTATION TEST")
    print("=" * 40)
    
    # Setup environment
    if not setup_test_environment():
        return
    
    # Check for image file
    if not check_image_file():
        save_sample_instructions()
        print("\n💡 NEXT STEPS:")
        print("1. Save your chest X-ray image as 'xray.jpg' in the current directory")
        print("2. Run this script again: python test_xray_analysis.py")
        print("3. Check XRAY_SETUP_INSTRUCTIONS.txt for detailed instructions")
        return
    
    # Run the analysis
    if run_anatomy_analysis():
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"📂 Check the 'xray_masks' folder for all results!")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main() 