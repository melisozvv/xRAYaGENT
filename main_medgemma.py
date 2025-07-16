#!/usr/bin/env python3
"""
Main script to process X-ray analysis using MedGemma VQA model
Processes 10 questions from questıons.csv with 500 data samples from selected_500_samples.json
"""

import os
import sys
import json
import csv
import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medgemma_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def json_serializer(obj):
    """Custom JSON serializer for numpy and other non-serializable objects"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def extract_and_parse_json(text: str) -> Union[Dict[str, Any], List[Any], str]:
    """
    Extract JSON from markdown code blocks and parse it into actual JSON objects.
    
    Args:
        text: Raw text that may contain JSON in markdown code blocks
        
    Returns:
        Parsed JSON object if valid JSON found, otherwise original text
    """
    # Pattern to match ```json ... ``` blocks
    json_pattern = r'```json\s*\n(.*?)\n```'
    
    # Try to find JSON in code blocks
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Take the first JSON block found
        json_str = matches[0].strip()
        try:
            # Parse the JSON
            parsed_json = json.loads(json_str)
            logger.debug(f"Successfully parsed JSON: {parsed_json}")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON block: {e}")
            logger.warning(f"JSON content: {json_str}")
    
    # If no JSON blocks found, try to extract JSON from the entire text
    # Look for patterns that start with { or [ (common JSON starts)
    json_pattern_loose = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
    
    matches = re.findall(json_pattern_loose, text)
    if matches:
        for match in matches:
            try:
                parsed_json = json.loads(match.strip())
                logger.debug(f"Successfully parsed loose JSON: {parsed_json}")
                return parsed_json
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found, return original text
    return text

class MedGemmaBatchProcessor:
    """Processes multiple X-ray images with multiple questions using MedGemma VQA"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the batch processor
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        
        # Initialize MedGemma pipeline
        logger.info("Initializing MedGemma pipeline...")
        self.pipe = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"MedGemma pipeline initialized on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Load questions and samples
        self.questions = self._load_questions()
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.questions)} questions and {len(self.samples)} samples")
    
    def _load_questions(self) -> List[str]:
        """Load questions from questıons.csv"""
        questions_file = self.data_dir / "questıons.csv"
        questions = []
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():  # Skip empty rows
                        questions.append(row[0].strip())
            
            logger.info(f"Loaded {len(questions)} questions from {questions_file}")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return []
    
    def _load_samples(self) -> Dict[str, Any]:
        """Load samples from selected_500_samples.json"""
        samples_file = self.data_dir / "selected_500_samples.json"
        
        try:
            with open(samples_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            logger.info(f"Loaded {len(samples)} samples from {samples_file}")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading samples: {e}")
            return {}
    
    def _get_valid_image_path(self, sample: Dict[str, Any]) -> Optional[str]:
        """Get the first valid image path for a sample"""
        image_paths = sample.get("ImagePath", [])
        
        for image_path in image_paths:
            # Convert relative path to use processed images
            processed_path = image_path.replace("../deid_png", "./processed_deid_png")
            full_path = self.data_dir / Path(processed_path)
            
            if full_path.exists():
                return str(full_path)
            else:
                logger.debug(f"Image path {full_path} does not exist")
        
        return None
    
    def process_sample_with_question(self, sample_id: str, sample: Dict[str, Any], 
                                   question: str, question_idx: int) -> Dict[str, Any]:
        """
        Process a single sample with a single question using MedGemma
        
        Args:
            sample_id: Unique identifier for the sample
            sample: Sample data dictionary
            question: Question to ask about the X-ray
            question_idx: Index of the question (0-based)
            
        Returns:
            Dictionary containing the analysis results
        """
        start_time = time.time()
        
        # Get valid image path
        image_path = self._get_valid_image_path(sample)
        if not image_path:
            return {
                "error": "No valid image path found",
                "sample_id": sample_id,
                "question": question,
                "question_idx": question_idx,
                "processed_at": datetime.now().isoformat()
            }
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Prepare messages for MedGemma
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist. Provide detailed, accurate answers to medical questions about chest X-ray images. Be precise and clinical in your assessment."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}, {"type": "image", "image": image}]
                }
            ]
            
            # Process with MedGemma
            logger.info(f"Processing sample {sample_id} with question {question_idx + 1}: {question[:50]}...")
            output_medgemma = self.pipe(text=messages, max_new_tokens=300)
            raw_answer = output_medgemma[0]["generated_text"][-1]["content"]
            
            # Extract and parse JSON if present
            answer = extract_and_parse_json(raw_answer)
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                "sample_id": sample_id,
                "question": question,
                "question_idx": question_idx,
                "answer": answer,
                "raw_answer": raw_answer,  # Keep original for debugging
                "answer_type": type(answer).__name__,  # Track if it's dict, list, or str
                "image_path": image_path,
                "processing_time_seconds": processing_time,
                "model": "google/medgemma-4b-it",
                "patient_id": sample.get("PatientID", ""),
                "study_date": sample.get("StudyDate", ""),
                "findings": sample.get("Findings", ""),
                "impression": sample.get("Impression", ""),
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Completed processing in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Error processing sample {sample_id} with question {question_idx + 1}: {e}")
            return {
                "error": str(e),
                "sample_id": sample_id,
                "question": question,
                "question_idx": question_idx,
                "image_path": image_path,
                "processing_time_seconds": processing_time,
                "processed_at": datetime.now().isoformat()
            }
    
    def process_all_samples(self, max_samples: Optional[int] = None, start_from: int = 0) -> Dict[str, Any]:
        """
        Process all samples with all questions using MedGemma, one at a time, and save immediately
        
        Args:
            max_samples: Maximum number of samples to process (None for all)
            start_from: Index to start processing from
            
        Returns:
            Dictionary with sample IDs as keys and their question results
        """
        # Load existing results to continue from where we left off
        structured_results = self.load_existing_results()
        
        sample_items = list(self.samples.items())
        
        # Apply limits
        if start_from > 0:
            sample_items = sample_items[start_from:]
        if max_samples:
            sample_items = sample_items[:max_samples]
        
        # Filter out already processed samples
        remaining_samples = [(sid, s) for sid, s in sample_items if sid not in structured_results]
        
        total_samples = len(sample_items)
        already_processed = len(sample_items) - len(remaining_samples)
        
        logger.info(f"🚀 Starting MedGemma batch processing")
        logger.info(f"Total samples to consider: {total_samples}")
        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Remaining to process: {len(remaining_samples)}")
        logger.info(f"Questions per sample: {len(self.questions)}")
        
        for sample_idx, (sample_id, sample) in enumerate(remaining_samples):
            logger.info(f"📋 Processing sample {sample_idx + 1}/{len(remaining_samples)}: {sample_id}")
            
            # Initialize sample results structure
            sample_results = {}
            
            for question_idx, question in enumerate(self.questions):
                logger.info(f"  ❓ Question {question_idx + 1}/{len(self.questions)}")
                
                # Process single sample-question combination
                result = self.process_sample_with_question(
                    sample_id, sample, question, question_idx
                )
                
                # Store result with question key (question1, question2, etc.)
                question_key = f"question{question_idx + 1}"
                sample_results[question_key] = result
                
                # Optional: Add small delay to avoid overwhelming GPU
                time.sleep(0.5)
            
            # Save this sample immediately
            self.save_single_sample(sample_id, sample_results)
            
            # Add to overall results
            structured_results[sample_id] = sample_results
            
            logger.info(f"✅ Completed sample {sample_idx + 1}/{len(remaining_samples)}: {sample_id}")
        
        logger.info(f"🎉 Completed processing. Total samples in results: {len(structured_results)}")
        return structured_results
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            filename = "medgemma_analysis_results.json"
        
        output_medgemma_dir = Path("/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_medgemma/medgemma")
        output_medgemma_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_medgemma_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        logger.info(f"Results saved to {filepath}")
        
        # Count total results
        total_results = sum(len(sample_results) for sample_results in results.values())
        logger.info(f"Total results: {total_results}")
        
        # Create summary statistics
        self._create_summary_stats(results, filepath.with_suffix('.summary.json'))
    
    def save_single_sample(self, sample_id: str, sample_results: Dict[str, Any]):
        """Save results for a single sample"""
        filename = f"medgemma_sample_{sample_id}.json"
        output_medgemma_dir = Path("/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_medgemma/medgemma")
        output_medgemma_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_medgemma_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        logger.info(f"Sample {sample_id} results saved to {filepath}")
    
    def load_existing_results(self) -> Dict[str, Any]:
        """Load existing results to continue processing"""
        output_medgemma_dir = Path("/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_medgemma/medgemma")
        results = {}
        
        # Look for existing sample files
        for sample_file in output_medgemma_dir.glob("medgemma_sample_*.json"):
            try:
                sample_id = sample_file.stem.replace("medgemma_sample_", "")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                results[sample_id] = sample_data
                logger.info(f"Loaded existing results for sample: {sample_id}")
            except Exception as e:
                logger.warning(f"Error loading {sample_file}: {e}")
        
        if results:
            logger.info(f"📂 Loaded existing results for {len(results)} samples")
        else:
            logger.info("🆕 No existing results found, starting fresh")
        
        return results
    
    def _create_summary_stats(self, results: Dict[str, Any], summary_filepath: Path):
        """Create summary statistics from results"""
        try:
            # Flatten results to count statistics
            flat_results = []
            total_processing_time = 0
            successful_results = []
            
            for sample_id, sample_results in results.items():
                for question_key, result in sample_results.items():
                    flat_results.append(result)
                    if "processing_time_seconds" in result:
                        total_processing_time += result["processing_time_seconds"]
                    if "error" not in result:
                        successful_results.append(result)
            
            # Count successful vs failed processing
            successful = len(successful_results)
            failed = len(flat_results) - successful
            
            # Count JSON parsing success
            json_parsed = sum(1 for r in successful_results if r.get("answer_type") in ["dict", "list"])
            text_answers = sum(1 for r in successful_results if r.get("answer_type") == "str")
            
            # Count by question
            question_stats = {}
            for result in flat_results:
                question = result.get("question", "Unknown")
                if question not in question_stats:
                    question_stats[question] = {
                        "successful": 0, 
                        "failed": 0, 
                        "avg_processing_time": 0,
                        "json_parsed": 0,
                        "text_answers": 0
                    }
                
                if "error" not in result:
                    question_stats[question]["successful"] += 1
                    if result.get("answer_type") in ["dict", "list"]:
                        question_stats[question]["json_parsed"] += 1
                    elif result.get("answer_type") == "str":
                        question_stats[question]["text_answers"] += 1
                else:
                    question_stats[question]["failed"] += 1
            
            # Calculate average processing times per question
            for question in question_stats:
                question_results = [r for r in flat_results if r.get("question") == question and "error" not in r]
                if question_results:
                    avg_time = sum(r.get("processing_time_seconds", 0) for r in question_results) / len(question_results)
                    question_stats[question]["avg_processing_time"] = avg_time
            
            summary = {
                "model": "google/medgemma-4b-it",
                "total_results": len(flat_results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(flat_results) if flat_results else 0,
                "json_parsing": {
                    "json_parsed": json_parsed,
                    "text_answers": text_answers,
                    "json_parse_rate": json_parsed / successful if successful else 0,
                    "total_successful": successful
                },
                "total_processing_time_seconds": total_processing_time,
                "average_processing_time_per_question": total_processing_time / len(flat_results) if flat_results else 0,
                "questions_processed": len(self.questions),
                "samples_processed": len(results),
                "question_statistics": question_stats,
                "structure_info": {
                    "format": "nested_by_sample",
                    "sample_keys": list(results.keys())[:5] if results else [],  # Show first 5 sample keys
                    "total_samples": len(results)
                },
                "device_used": "cuda" if torch.cuda.is_available() else "cpu",
                "generated_at": datetime.now().isoformat()
            }
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            logger.info(f"📊 Summary statistics saved to {summary_filepath}")
            logger.info(f"📈 Success rate: {summary['success_rate']:.2%}")
            logger.info(f"🔧 JSON parse rate: {summary['json_parsing']['json_parse_rate']:.2%} ({json_parsed}/{successful} successful answers)")
            logger.info(f"⏱️  Average processing time: {summary['average_processing_time_per_question']:.2f}s per question")
            
        except Exception as e:
            logger.error(f"Error creating summary statistics: {e}")

def main():
    """Main function to run the MedGemma batch processing"""
    logger.info("🚀 Starting MedGemma X-ray batch processing")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.info("💻 Using CPU (consider using GPU for faster processing)")
    
    # Initialize processor
    processor = MedGemmaBatchProcessor()
    
    # Check if questions and samples loaded successfully
    if not processor.questions:
        logger.error("❌ No questions loaded. Exiting.")
        return
    
    if not processor.samples:
        logger.error("❌ No samples loaded. Exiting.")
        return
    
    # Process all samples (will automatically continue from where it left off)
    results = processor.process_all_samples()
    
    # Save consolidated results file
    processor.save_results(results, "medgemma_analysis_results.json")
    
    logger.info("🎉 MedGemma batch processing completed successfully")
    logger.info(f"📁 Individual sample files: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_medgemma/medgemma/medgemma_sample_*.json")
    logger.info(f"📄 Consolidated results: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_medgemma/medgemma/medgemma_analysis_results.json")

if __name__ == "__main__":
    main() 