#!/usr/bin/env python3
"""
Main script to process X-ray analysis using XrayAgent
Processes 10 questions from questıons.csv with 500 data samples from selected_500_samples.json
"""

import os
import sys
import json
import csv
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the XrayAgent code
from xray_agent import XrayAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xray_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XrayBatchProcessor:
    """Processes multiple X-ray images with multiple questions using XrayAgent"""
    
    def __init__(self, data_dir: str = "data", tools_dir: str = "src/tools"):
        """
        Initialize the batch processor
        
        Args:
            data_dir: Directory containing the data files
            tools_dir: Directory containing the analysis tools
        """
        self.data_dir = Path(data_dir)
        self.tools_dir = tools_dir
        self.xray_agent = XrayAgent(tools_dir=tools_dir)
        
        # Load questions and samples
        self.questions = self._load_questions()
        self.questions = self.questions[:1]
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
            processed_path = image_path.replace("../deid_png", "processed_deid_png")
            full_path = self.data_dir / processed_path
            
            if full_path.exists():
                return str(full_path)
        
        return None
    
    def process_sample_with_question(self, sample_id: str, sample: Dict[str, Any], 
                                   question: str, question_idx: int) -> Dict[str, Any]:
        """
        Process a single sample with a single question
        
        Args:
            sample_id: Unique identifier for the sample
            sample: Sample data dictionary
            question: Question to ask about the X-ray
            question_idx: Index of the question (0-based)
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            # Get valid image path
            image_path = self._get_valid_image_path(sample)
            if not image_path:
                return {
                    "error": "No valid image path found",
                    "sample_id": sample_id,
                    "question": question,
                    "question_idx": question_idx
                }
            
            # Process the query using XrayAgent
            logger.info(f"Processing sample {sample_id} with question {question_idx + 1}")
            result = self.xray_agent.process_query(image_path, question)
            
            # Add metadata
            result.update({
                "sample_id": sample_id,
                "question": question,
                "question_idx": question_idx,
                "patient_id": sample.get("PatientID", ""),
                "study_date": sample.get("StudyDate", ""),
                "findings": sample.get("Findings", ""),
                "impression": sample.get("Impression", ""),
                "processed_at": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_id} with question {question_idx}: {e}")
            return {
                "error": str(e),
                "sample_id": sample_id,
                "question": question,
                "question_idx": question_idx
            }
    
    def process_all_samples(self, max_samples: Optional[int] = None, start_from: int = 0) -> Dict[str, Any]:
        """
        Process all samples with all questions and return structured results
        
        Args:
            max_samples: Maximum number of samples to process (None for all)
            start_from: Index to start processing from
            
        Returns:
            Dictionary with sample IDs as keys and their question results
        """
        structured_results = {}
        sample_items = list(self.samples.items())
        sample_items = sample_items[:1]
        
        # Apply limits
        if start_from > 0:
            sample_items = sample_items[start_from:]
        if max_samples:
            sample_items = sample_items[:max_samples]
        
        total_tasks = len(sample_items) * len(self.questions)
        current_task = 0
        
        logger.info(f"Processing {len(sample_items)} samples with {len(self.questions)} questions each")
        logger.info(f"Total tasks: {total_tasks}")
        
        for sample_idx, (sample_id, sample) in enumerate(sample_items):
            logger.info(f"Processing sample {sample_idx + 1}/{len(sample_items)}: {sample_id}")
            
            # Initialize sample results structure
            sample_results = {}
            
            for question_idx, question in enumerate(self.questions):
                current_task += 1
                
                # Process single sample-question combination
                result = self.process_sample_with_question(
                    sample_id, sample, question, question_idx
                )
                
                # Store result with question key (question1, question2, etc.)
                question_key = f"question{question_idx + 1}"
                sample_results[question_key] = result
                
                # Log progress
                if current_task % 10 == 0:
                    logger.info(f"Completed {current_task}/{total_tasks} tasks ({current_task/total_tasks*100:.1f}%)")
                
                # Optional: Add small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            # Add all results for this sample
            structured_results[sample_id] = sample_results
            
            # Save intermediate results every 10 samples
            if (sample_idx + 1) % 10 == 0:
                self._save_intermediate_results(structured_results, sample_idx + 1)
        
        logger.info(f"Completed processing all {len(sample_items)} samples")
        return structured_results
    
    def _save_intermediate_results(self, results: Dict[str, Any], sample_count: int):
        """Save intermediate results to avoid data loss"""
        filename = f"intermediate_results_{sample_count}_samples.json"
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results to {filepath}")
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xray_analysis_results_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            
            # Count total results
            total_results = sum(len(sample_results) for sample_results in results.values())
            logger.info(f"Total results: {total_results}")
            
            # Create summary statistics
            self._create_summary_stats(results, filepath.with_suffix('.summary.json'))
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _create_summary_stats(self, results: Dict[str, Any], summary_filepath: Path):
        """Create summary statistics from results"""
        try:
            # Flatten results to count statistics
            flat_results = []
            for sample_id, sample_results in results.items():
                for question_key, result in sample_results.items():
                    flat_results.append(result)
            
            # Count successful vs failed processing
            successful = sum(1 for r in flat_results if "error" not in r)
            failed = len(flat_results) - successful
            
            # Count by question
            question_stats = {}
            for result in flat_results:
                question = result.get("question", "Unknown")
                if question not in question_stats:
                    question_stats[question] = {"successful": 0, "failed": 0}
                
                if "error" not in result:
                    question_stats[question]["successful"] += 1
                else:
                    question_stats[question]["failed"] += 1
            
            summary = {
                "total_results": len(flat_results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(flat_results) if flat_results else 0,
                "questions_processed": len(self.questions),
                "samples_processed": len(results),
                "question_statistics": question_stats,
                "structure_info": {
                    "format": "nested_by_sample",
                    "sample_keys": list(results.keys())[:5] if results else [],  # Show first 5 sample keys
                    "total_samples": len(results)
                },
                "generated_at": datetime.now().isoformat()
            }
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary statistics saved to {summary_filepath}")
            
        except Exception as e:
            logger.error(f"Error creating summary statistics: {e}")

def main():
    """Main function to run the batch processing"""
    logger.info("Starting X-ray batch processing")
    
    # Initialize processor
    processor = XrayBatchProcessor()
    
    # Check if questions and samples loaded successfully
    if not processor.questions:
        logger.error("No questions loaded. Exiting.")
        return
    
    if not processor.samples:
        logger.error("No samples loaded. Exiting.")
        return
    
    # Process all samples (you can limit this for testing)
    # For testing, you might want to use: max_samples=10
    results = processor.process_all_samples(max_samples=None)  # Process all 500 samples
    
    # Save results
    processor.save_results(results)
    
    logger.info("Batch processing completed successfully")

if __name__ == "__main__":
    main() 