#!/usr/bin/env python3
"""
Image processing script to convert 16-bit X-ray images to 8-bit format
Creates processed_deid_png directory with normalized images (0-255 range)
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_image_to_8bit(image_path: Path, output_path: Path):
    """
    Convert a 16-bit image to 8-bit format (0-255 range)
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Get original data type and range
        original_dtype = img_array.dtype
        original_max = np.max(img_array)
        original_min = np.min(img_array)
        
        logger.debug(f"Processing {image_path.name}: {original_dtype}, range [{original_min}, {original_max}]")
        
        # Normalize to 0-255 range
        if original_max > 255:
            # Scale from current range to 0-255
            img_normalized = ((img_array - original_min) / (original_max - original_min) * 255).astype(np.uint8)
        else:
            # Already in 8-bit range, just ensure uint8 type
            img_normalized = img_array.astype(np.uint8)
        
        # Create PIL image and save
        processed_img = Image.fromarray(img_normalized)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG
        processed_img.save(output_path, 'PNG')
        
        logger.debug(f"Saved normalized image: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def process_directory(input_dir: Path, output_dir: Path):
    """
    Process all PNG images in input directory and save to output directory
    
    Args:
        input_dir: Source directory with original images
        output_dir: Destination directory for processed images
    """
    
    # Find all PNG files
    png_files = list(input_dir.rglob("*.png"))
    
    if not png_files:
        logger.warning(f"No PNG files found in {input_dir}")
        return
    
    logger.info(f"Found {len(png_files)} PNG files to process")
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(png_files, desc="Processing images"):
        # Calculate relative path to maintain directory structure
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        # Process the image
        if normalize_image_to_8bit(img_path, output_path):
            success_count += 1
        else:
            error_count += 1
    
    logger.info(f"Processing complete: {success_count} successful, {error_count} errors")

def main():
    """Main function to process all images"""
    
    # Define directories
    input_dir = Path("data/deid_png")
    output_dir = Path("data/processed_deid_png")
    
    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    if output_dir.exists():
        logger.info(f"Output directory already exists: {output_dir}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            logger.info("Processing cancelled")
            return
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Process all images
    process_directory(input_dir, output_dir)
    
    logger.info("Image processing completed!")

if __name__ == "__main__":
    main() 