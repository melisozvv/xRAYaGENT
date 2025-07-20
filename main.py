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
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the XrayAgent code
from xray_agent_google import XrayAgent

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

class XrayBatchProcessor:
    """Processes multiple X-ray images with multiple questions using XrayAgent"""
    
    def __init__(self, data_dir: str = "/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/data", tools_dir: str = "src/tools"):
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
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.questions)} questions and {len(self.samples)} samples")
    
    def _load_questions(self) -> List[str]:
        """Load questions from questıons.csv"""
        questions = ['"Is there any evidence of COVID-19? RESPONSE IN JSON WITH {""EXIST"": 0 or 1, ""BOUNDING_BOX"": [x_topleft, y_topleft, x_bottomright, y_bottomright], ""LOCATION"": ""left lung"" or ""right lung"" or ""lung upper lobe left"" or ""lung upper lobe right"" or ""lung lower lobe left"" or ""lung lower lobe right"" or ""lung middle lobe right""}"']
        return questions
        # questions_file = self.data_dir / "questıons.csv"
        # questions = []
        
        # try:
        #     with open(questions_file, 'r', encoding='utf-8') as f:
        #         reader = csv.reader(f)
        #         for row in reader:
        #             if row and row[0].strip():  # Skip empty rows
        #                 questions.append(row[0].strip())
     
            
        #     logger.info(f"Loaded {len(questions)} questions from {questions_file}")
        #     return questions
            
        # except Exception as e:
        #     logger.error(f"Error loading questions: {e}")
        #     return []
    
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
                print(f"Image path {full_path} exists")
                return str(full_path)
            else:
                print(f"Image path {full_path} does not exist")
        
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
        result = self.xray_agent.process_query(image_path, question, study_id=sample_id, question_id=str(question_idx + 1))
        
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
    
    def process_all_samples(self, max_samples: Optional[int] = None, start_from: int = 0) -> Dict[str, Any]:
        """
        Process all samples with all questions, one at a time, and save immediately
        
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
        
        logger.info(f"Total samples to consider: {total_samples}")
        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Remaining to process: {len(remaining_samples)}")
        logger.info(f"Questions per sample: {len(self.questions)}")
        
        for sample_idx, (sample_id, sample) in enumerate(remaining_samples):
            logger.info(f"Processing sample {sample_idx + 1}/{len(remaining_samples)}: {sample_id}")
            
            # Initialize sample results structure
            sample_results = {}
            
            for question_idx, question in enumerate(self.questions):
                logger.info(f"  Question {question_idx + 1}/{len(self.questions)}: {question[:50]}...")
                
                # Process single sample-question combination
                result = self.process_sample_with_question(
                    sample_id, sample, question, question_idx
                )
                
                # Store result with question key (question1, question2, etc.)
                question_key = f"question{question_idx + 1}"
                sample_results[question_key] = result
                
                # Optional: Add small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            # Save this sample immediately
            self.save_single_sample(sample_id, sample_results)
            
            # Add to overall results
            structured_results[sample_id] = sample_results
            
            logger.info(f"✅ Completed sample {sample_idx + 1}/{len(remaining_samples)}: {sample_id}")
        
        logger.info(f"Completed processing. Total samples in results: {len(structured_results)}")
        return structured_results
    
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            filename = "xray_analysis_results.json"
        
        output_dir = Path("/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_google")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        
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
        filename = f"sample_{sample_id}.json"
        output_dir = Path("/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        logger.info(f"Sample {sample_id} results saved to {filepath}")
    
    def load_existing_results(self) -> Dict[str, Any]:
        """Load existing results to continue processing"""
        output_dir = Path("/home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_google")
        results = {}
        
        # Look for existing sample files
        for sample_file in output_dir.glob("sample_*.json"):
            try:
                sample_id = sample_file.stem.replace("sample_", "")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                results[sample_id] = sample_data
                logger.info(f"Loaded existing results for sample: {sample_id}")
            except Exception as e:
                logger.warning(f"Error loading {sample_file}: {e}")
        
        if results:
            logger.info(f"Loaded existing results for {len(results)} samples")
        else:
            logger.info("No existing results found, starting fresh")
        
        return results
            
    
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
                json.dump(summary, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            logger.info(f"Summary statistics saved to {summary_filepath}")
            
        except Exception as e:
            logger.error(f"Error creating summary statistics: {e}")

def main():
    """Main function to run the batch processing"""
    logger.info("Starting X-ray batch processing (one sample at a time)")
    
    # Initialize processor
    processor = XrayBatchProcessor()
    
    # Check if questions and samples loaded successfully
    if not processor.questions:
        logger.error("No questions loaded. Exiting.")
        return
    
    if not processor.samples:
        logger.error("No samples loaded. Exiting.")
        return
    
    # Process all samples (will automatically continue from where it left off)
    results = processor.process_all_samples()
    
    # Save consolidated results file
    processor.save_results(results, "xray_analysis_results.json")
    
    logger.info("Batch processing completed successfully")
    logger.info(f"Individual sample files: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_google/sample_*.json")
    logger.info(f"Consolidated results: /home/xiz569/rajpurkarlab/home/xiz569/xRAYaGENT/output_google/xray_analysis_results.json")

if __name__ == "__main__":
    main() 