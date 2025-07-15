#!/usr/bin/env python3
"""
Fix for CXAS PyTorch 2.6+ compatibility issue
This script patches the torch.load issue with weights_only=True default
"""

import torch
import argparse
import os
import sys

def patch_torch_load():
    """Patch torch.load to allow argparse.Namespace as safe global"""
    # Add argparse.Namespace to safe globals
    torch.serialization.add_safe_globals([argparse.Namespace])
    print("✅ Added argparse.Namespace to PyTorch safe globals")

def run_cxas_segment(input_path, output_path, output_type="png", gpus="cpu"):
    """
    Run CXAS segmentation with the torch patch applied
    """
    try:
        # Apply the patch first
        patch_torch_load()
        
        # Import and use CXAS after patching
        from cxas.segmentor import CXAS
        
        print(f"🔬 Running CXAS segmentation...")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        print(f"   Format: {output_type}")
        print(f"   Device: {gpus}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize CXAS with default model
        model = CXAS(gpus=gpus)
        
        # Run segmentation
        result = model.segment(input_path, output_path, output_type=output_type)
        
        print("✅ CXAS segmentation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CXAS segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test the patched CXAS"""
    parser = argparse.ArgumentParser(description='Run CXAS with PyTorch 2.6+ compatibility fix')
    parser.add_argument('-i', '--input', required=True, help='Input chest X-ray image path')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    parser.add_argument('-ot', '--output_type', default='png', help='Output format (default: png)')
    parser.add_argument('-g', '--gpus', default='cpu', help='GPU device (default: cpu)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found!")
        return 1
    
    # Run the patched CXAS segmentation
    success = run_cxas_segment(args.input, args.output, args.output_type, args.gpus)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 