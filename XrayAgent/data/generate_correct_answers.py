#!/usr/bin/env python3
"""
Generate correct answers to X-ray questions based on findings and impression data
Uses GPT-4.1 for accurate medical interpretation
"""

import json
import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import Azure OpenAI
try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError("OpenAI library is required. Install with: pip install openai>=1.0.0")

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-ai.hms.edu")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1")
AZURE_OPENAI_API_KEY = "b960432f5dd540969d3083910b085a33"

def get_azure_client():
    """Initialize and return Azure OpenAI client"""
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("Please set AZURE_OPENAI_API_KEY environment variable")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

class GPTBasedMedicalAnalyzer:
    """Use GPT-4.1 to analyze medical findings and impressions"""
    
    def __init__(self):
        self.client = get_azure_client()
        self.system_prompt = """You are an expert radiologist analyzing chest X-ray findings and impressions. 
        You will be provided with the medical findings and impression from a chest X-ray report, along with a specific question.
        
        Your task is to analyze the provided medical text and answer the question accurately based on the clinical information.
        
        Important guidelines:
        1. Base your answers ONLY on the information provided in the findings and impression
        2. If the information is not explicitly mentioned or cannot be reasonably inferred, answer negatively
        3. Use standard medical terminology and interpretations
        4. Provide answers in the exact JSON format requested
        5. Be conservative - only answer positively if there's clear evidence in the text
        
        Remember: You are analyzing existing radiology reports, not the actual images."""
    
    def analyze_with_gpt(self, findings: str, impression: str, question: str) -> Dict[str, Any]:
        """Use GPT-4.1 to analyze findings and impression for a specific question"""
        
        user_prompt = f"""
Medical Findings: {findings or 'No findings provided'}
Medical Impression: {impression or 'No impression provided'}

Question: {question}

Please analyze the above medical findings and impression to answer the question. 
Provide your answer in the exact JSON format specified in the question.
Be precise and base your answer only on the information provided in the findings and impression.
If the information is not mentioned or cannot be reasonably inferred, answer negatively.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
                else:
                    # Fallback: try to parse the entire response
                    return json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a default response based on question type
                return self._create_default_response(question)
                
        except Exception as e:
            print(f"Error with GPT analysis: {e}")
            return self._create_default_response(question)
    
    def _create_default_response(self, question: str) -> Dict[str, Any]:
        """Create a default response when GPT analysis fails"""
        question_lower = question.lower()
        
        if "tuberculosis" in question_lower:
            return {"EXIST": 0}
        elif "et tube" in question_lower:
            return {"EXIST": 0, "DISTANCE": 0.0}
        elif "nodule" in question_lower:
            return {"EXIST": 0, "LOCATION": ""}
        elif "fracture" in question_lower:
            return {"EXIST": 0, "LOCATION": "", "BOUNDING_BOX": []}
        elif "mediastinum" in question_lower:
            return {"NORMAL": 1, "FINDINGS": "Unable to analyze"}
        elif "left lung" in question_lower:
            return {"LEFT_LUNG": []}
        elif "right lung" in question_lower:
            return {"RIGHT_LUNG": []}
        elif "pneumothorax" in question_lower:
            return {"EXIST": 0, "LOCATION": ""}
        elif "pleural effusion" in question_lower:
            return {"EXIST": 0, "LOCATION": ""}
        elif "pneumonia" in question_lower:
            return {"EXIST": 0, "LOCATION": ""}
        elif "hilar lymphadenopathy" in question_lower:
            return {"EXIST": 0, "LOCATION": ""}
        else:
            return {"EXIST": 0}

def generate_correct_answers():
    """Generate correct answers for all samples and questions using GPT-4.1"""
    
    # Load questions
    questions_file = Path("questıons.csv")
    questions = []
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                questions.append(row[0].strip())
    
    # Load samples
    samples_file = Path("selected_500_samples.json")
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Initialize analyzer
    analyzer = GPTBasedMedicalAnalyzer()
    
    # Generate answers for first 10 samples (for testing)
    structured_results = {}
    sample_items = list(samples.items())[:10]  # Limit to 10 samples for testing
    
    print(f"Processing {len(sample_items)} samples using GPT-4.1...")
    
    for idx, (sample_id, sample_data) in enumerate(sample_items, 1):
        print(f"Processing sample {idx}/{len(sample_items)}: {sample_id[:30]}...")
        
        findings = sample_data.get("Findings", "")
        impression = sample_data.get("Impression", "")
        image_paths = sample_data.get("ImagePath", [])
        
        # Use first image path, convert to processed format
        if image_paths:
            image_path = image_paths[0].replace("../deid_png", "processed_deid_png")
        else:
            image_path = ""
        
        sample_results = {}
        
        for question_idx, question in enumerate(questions):
            question_key = f"question{question_idx + 1}"
            
            print(f"  Analyzing question {question_idx + 1}...")
            
            # Use GPT-4.1 to analyze the question
            answer = analyzer.analyze_with_gpt(findings, impression, question)
            
            # Create result structure
            result = {
                "query": question,
                "image_path": image_path,
                "results": {
                    "answer": answer
                }
            }
            
            sample_results[question_key] = result
        
        structured_results[sample_id] = sample_results
        
        # Add small delay to avoid rate limiting
        import time
        time.sleep(0.5)
    
    return structured_results

def generate_all_correct_answers():
    """Generate correct answers for ALL samples using GPT-4.1"""
    
    # Load questions
    questions_file = Path("questıons.csv")
    questions = []
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                questions.append(row[0].strip())
    
    # Load samples
    samples_file = Path("selected_500_samples.json")
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Initialize analyzer
    analyzer = GPTBasedMedicalAnalyzer()
    
    # Generate answers for ALL samples
    structured_results = {}
    
    print(f"Processing ALL {len(samples)} samples using GPT-4.1...")
    print("This will take a significant amount of time due to API calls...")
    
    for idx, (sample_id, sample_data) in enumerate(samples.items(), 1):
        if idx % 10 == 0:
            print(f"Processing sample {idx}/{len(samples)}: {sample_id[:30]}...")
        
        findings = sample_data.get("Findings", "")
        impression = sample_data.get("Impression", "")
        image_paths = sample_data.get("ImagePath", [])
        
        # Use first image path, convert to processed format
        if image_paths:
            image_path = image_paths[0].replace("../deid_png", "processed_deid_png")
        else:
            image_path = ""
        
        sample_results = {}
        
        for question_idx, question in enumerate(questions):
            question_key = f"question{question_idx + 1}"
            
            # Use GPT-4.1 to analyze the question
            answer = analyzer.analyze_with_gpt(findings, impression, question)
            
            # Create result structure
            result = {
                "query": question,
                "image_path": image_path,
                "results": {
                    "answer": answer
                }
            }
            
            sample_results[question_key] = result
        
        structured_results[sample_id] = sample_results
        
        # Add small delay to avoid rate limiting
        import time
        time.sleep(0.1)
    
    return structured_results

def main():
    """Main function to generate correct answers"""
    print("=" * 60)
    print("GPT-4.1 Based Correct Answer Generation")
    print("=" * 60)
    print()
    
    # Ask user which version to run
    print("Choose processing option:")
    print("1. Test with 10 samples (recommended for initial testing)")
    print("2. Process ALL 500 samples (will take significant time)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nGenerating correct answers for 10 samples using GPT-4.1...")
        results = generate_correct_answers()
        suffix = "test_10_samples"
    elif choice == "2":
        print("\nGenerating correct answers for ALL 500 samples using GPT-4.1...")
        print("This will take a long time and use many API calls...")
        confirm = input("Are you sure you want to proceed? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
        results = generate_all_correct_answers()
        suffix = "all_500_samples"
    else:
        print("Invalid choice. Defaulting to test mode with 10 samples.")
        results = generate_correct_answers()
        suffix = "test_10_samples"
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"gpt4_correct_answers_{suffix}_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCorrect answers saved to {output_file}")
    print(f"Total samples processed: {len(results)}")
    
    # Print summary
    if results:
        sample_id = list(results.keys())[0]
        print(f"Sample structure: {len(results[sample_id])} questions per sample")
        print(f"Question keys: {list(results[sample_id].keys())}")
        
        # Show first question result as example
        first_question = results[sample_id]["question1"]
        print(f"\nExample result for first question:")
        print(f"Query: {first_question['query'][:50]}...")
        print(f"Answer: {first_question['results']['answer']}")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = main()
        print(f"\n✅ GPT-4.1 based correct answers generated successfully!")
        print(f"📁 File saved as: {output_file}") 
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure AZURE_OPENAI_API_KEY environment variable is set.") 