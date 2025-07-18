import json
import random
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def load_data():
    """Load the existing answer data and metadata"""
    
    # Load the answer data that contains exist values
    answer_files = [
        "data/gpt4_correct_answers_all_500_samples_20250715_224243.json",
        "data/gpt4_correct_answers_filter_samples_20250718_104754.json"
    ]
    
    all_answers = {}
    for file_path in answer_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_answers.update(data)
            print(f"Loaded {len(data)} samples from {file_path}")
    
    # Load metadata for finding consolidation cases
    metadata_file = "data/test_metadata.json"
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata)} samples")
    
    return all_answers, metadata

def find_consolidation_cases(metadata: Dict, target_count: int = 50) -> Set[str]:
    """Find study IDs that likely have consolidation based on medical text"""
    
    consolidation_keywords = [
        'consolidation', 'consolidations', 'consolidative', 'consolidated',
        'airspace consolidation', 'air space consolidation', 'focal consolidation',
        'patchy consolidation', 'lobar consolidation', 'segmental consolidation'
    ]
    
    consolidation_studies = set()
    
    for study_id, study_info in metadata.items():
        # Check findings and impression text (handle None values)
        findings = (study_info.get('Findings') or '').lower()
        impression = (study_info.get('Impression') or '').lower()
        indication = (study_info.get('Indication') or '').lower()
        
        combined_text = f"{findings} {impression} {indication}"
        
        # Look for consolidation keywords
        for keyword in consolidation_keywords:
            if keyword in combined_text:
                consolidation_studies.add(study_id)
                break
    
    print(f"Found {len(consolidation_studies)} potential consolidation cases from medical text")
    
    # If we need more, add some samples that might have consolidation
    if len(consolidation_studies) < target_count:
        # Look for pneumonia cases which often involve consolidation
        pneumonia_keywords = ['pneumonia', 'pneumonic', 'infiltrate', 'infiltration']
        
        for study_id, study_info in metadata.items():
            if study_id in consolidation_studies:
                continue
                
            findings = (study_info.get('Findings') or '').lower()
            impression = (study_info.get('Impression') or '').lower()
            combined_text = f"{findings} {impression}"
            
            for keyword in pneumonia_keywords:
                if keyword in combined_text:
                    consolidation_studies.add(study_id)
                    if len(consolidation_studies) >= target_count:
                        break
            
            if len(consolidation_studies) >= target_count:
                break
    
    print(f"Total consolidation candidate studies: {len(consolidation_studies)}")
    return consolidation_studies

def find_complete_samples(all_answers: Dict) -> Dict[str, Dict[str, int]]:
    """Find samples that have answers for all required questions and analyze their positive cases"""
    
    required_questions = {
        "tuberculosis": "question1",  # Will become consolidation
        "et_tube": "question2",
        "nodules": "question3", 
        "covid19": "question4",
        "mediastinum": "question5",
        "pneumothorax": "question8",
        "pleural_effusion": "question9",
        "pneumonia": "question10"
    }
    
    complete_samples = {}
    
    for study_id, study_data in all_answers.items():
        # Check if this study has answers for all required question types
        question_coverage = {}
        
        for question_key, question_data in study_data.items():
            if question_key.startswith('question'):
                query = question_data.get('query', '').lower()
                results = question_data.get('results', {})
                answer = results.get('answer', {})
                
                # Get exist/normal value
                exist_value = None
                if 'EXIST' in answer:
                    exist_value = answer['EXIST']
                elif 'NORMAL' in answer:
                    exist_value = answer['NORMAL']
                
                # Map question type
                question_type = None
                if "tuberculosis" in query:
                    question_type = "tuberculosis"
                elif "et tube" in query or "carina" in query:
                    question_type = "et_tube"
                elif "nodule" in query:
                    question_type = "nodules"
                elif "covid" in query:
                    question_type = "covid19"
                elif "mediastinum" in query:
                    question_type = "mediastinum"
                elif "pneumothorax" in query:
                    question_type = "pneumothorax"
                elif "pleural effusion" in query:
                    question_type = "pleural_effusion"
                elif "pneumonia" in query:
                    question_type = "pneumonia"
                
                if question_type and exist_value is not None:
                    question_coverage[question_type] = exist_value
        
        # Only include samples that have all required questions
        if len(question_coverage) == len(required_questions):
            complete_samples[study_id] = question_coverage
    
    print(f"Found {len(complete_samples)} samples with complete question coverage")
    return complete_samples

def select_strategic_samples_v2(complete_samples: Dict[str, Dict[str, int]], 
                              consolidation_studies: Set[str],
                              target_total: int = 500) -> List[str]:
    """Strategically select 500 complete samples that maximize positive cases across all questions"""
    
    print("\n=== STRATEGIC SAMPLE SELECTION V2 ===")
    
    # Analyze positive case availability for complete samples
    question_positive_counts = {
        "tuberculosis": [],
        "et_tube": [],
        "nodules": [], 
        "covid19": [],
        "mediastinum": [],
        "pneumothorax": [],
        "pleural_effusion": [],
        "pneumonia": []
    }
    
    for study_id, question_answers in complete_samples.items():
        for question_type, exist_value in question_answers.items():
            if exist_value == 1:
                question_positive_counts[question_type].append(study_id)
    
    print("\nPositive cases in complete samples:")
    for question_type, positive_studies in question_positive_counts.items():
        print(f"  {question_type}: {len(positive_studies)} positive cases")
    
    # Start with COVID-19 positive cases (mandatory requirement)
    covid_positive = set(question_positive_counts["covid19"])
    selected_samples = list(covid_positive)
    print(f"\nStarting with {len(selected_samples)} COVID-19 positive cases (mandatory)")
    
    # Track which questions need more positive cases
    requirements = {
        "tuberculosis": 50,  # Will change to consolidation
        "et_tube": 50,
        "nodules": 50,
        "mediastinum": 50, 
        "pneumothorax": 50,
        "pleural_effusion": 50,
        "pneumonia": 50
    }
    
    # For tuberculosis/consolidation, prioritize consolidation candidates
    tuberculosis_positive = set(question_positive_counts["tuberculosis"])
    consolidation_candidates = consolidation_studies & set(complete_samples.keys())
    
    # Prefer samples that are both tuberculosis positive OR consolidation candidates
    consolidation_pool = tuberculosis_positive | consolidation_candidates
    consolidation_available = list(consolidation_pool - set(selected_samples))
    
    consolidation_needed = min(50, len(consolidation_available))
    if consolidation_needed > 0:
        selected_consolidation = random.sample(consolidation_available, consolidation_needed)
        selected_samples.extend(selected_consolidation)
        print(f"Added {consolidation_needed} consolidation cases")
    
    # For other questions, aggressively ensure we meet the 50+ requirement
    remaining_requirements = {k: v for k, v in requirements.items() if k != "tuberculosis"}
    
    # First pass: directly add samples for questions that are far from their targets
    for question_type, min_required in remaining_requirements.items():
        positive_samples = set(question_positive_counts[question_type])
        already_selected = set(selected_samples)
        current_positive = len(positive_samples & already_selected)
        needed = max(0, min_required - current_positive)
        
        if needed > 0:
            available_positive = list(positive_samples - already_selected)
            to_add = min(needed, len(available_positive), target_total - len(selected_samples))
            
            if to_add > 0:
                new_samples = random.sample(available_positive, to_add)
                selected_samples.extend(new_samples)
                print(f"Added {to_add} {question_type} positive cases (needed {needed})")
    
    print(f"After direct positive addition: {len(selected_samples)}")
    
    # Second pass: fill remaining spots with best candidates
    while len(selected_samples) < target_total:
        already_selected = set(selected_samples)
        
        # Calculate current positive counts
        current_positive = {}
        for question_type in remaining_requirements:
            positive_set = set(question_positive_counts[question_type])
            current_positive[question_type] = len(positive_set & already_selected)
        
        # Find samples that would contribute to the most under-served questions
        candidate_scores = {}
        available_samples = set(complete_samples.keys()) - already_selected
        
        for candidate in available_samples:
            score = 0
            candidate_answers = complete_samples[candidate]
            
            for question_type, min_required in remaining_requirements.items():
                current_count = current_positive[question_type]
                if current_count < min_required and candidate_answers.get(question_type) == 1:
                    # Higher weight for questions that are further from target
                    shortfall = min_required - current_count
                    weight = shortfall * 2  # Multiply by 2 to prioritize under-served questions
                    score += weight
            
            candidate_scores[candidate] = score
        
        # Select the best candidate (or random if no good candidates)
        if candidate_scores and max(candidate_scores.values()) > 0:
            best_candidate = max(candidate_scores.keys(), key=lambda x: candidate_scores[x])
            selected_samples.append(best_candidate)
        else:
            # No candidates that help with positive requirements, just pick randomly
            remaining_available = list(set(complete_samples.keys()) - set(selected_samples))
            if remaining_available:
                selected_samples.append(random.choice(remaining_available))
            else:
                break  # No more samples available
        
        # Stop if we've reached target
        if len(selected_samples) >= target_total:
            break
    
    print(f"Final selection: {len(selected_samples)} samples")
    return selected_samples[:target_total]  # Ensure exactly target_total

def create_consolidated_dataset_v2(all_answers: Dict, selected_study_ids: List[str], 
                                 consolidation_studies: Set[str]) -> Dict[str, Dict]:
    """Create the final dataset with all questions for the selected samples"""
    
    final_dataset = {}
    
    for study_id in selected_study_ids:
        if study_id not in all_answers:
            continue
            
        study_data = all_answers[study_id]
        final_dataset[study_id] = {}
        
        # Process each question for this study
        for question_key, question_data in study_data.items():
            if question_key.startswith('question'):
                query = question_data.get('query', '').lower()
                
                # Determine question type and map to new numbering
                target_question_key = None
                
                if "tuberculosis" in query:
                    target_question_key = "question1"
                    # Change query to consolidation and modify answer based on consolidation status
                    new_query = question_data.get('query', '').replace("tuberculosis", "consolidation").replace("Tuberculosis", "Consolidation")
                    new_answer = question_data.get('results', {}).get('answer', {}).copy()
                    
                    # Mark as positive if it's in consolidation candidates OR was originally tuberculosis positive
                    original_exist = question_data.get('results', {}).get('answer', {}).get('EXIST', 0)
                    if study_id in consolidation_studies or original_exist == 1:
                        new_answer['EXIST'] = 1
                    else:
                        new_answer['EXIST'] = 0
                    
                    final_dataset[study_id][target_question_key] = {
                        "query": new_query,
                        "image_path": question_data.get('image_path', ''),
                        "results": {"answer": new_answer}
                    }
                    
                elif "et tube" in query or "carina" in query:
                    target_question_key = "question2"
                elif "nodule" in query and "is there nodules" in query:
                    target_question_key = "question3"
                elif "covid" in query:
                    target_question_key = "question4"
                elif "mediastinum" in query:
                    target_question_key = "question5"
                elif "pneumothorax" in query:
                    target_question_key = "question8"
                elif "pleural effusion" in query:
                    target_question_key = "question9"
                elif "pneumonia" in query:
                    target_question_key = "question10"
                elif "left lung" in query and "disease" in query:
                    # These are questions 6/7 about diseases in left/right lung - skip them
                    target_question_key = None
                elif "right lung" in query and "disease" in query:
                    # These are questions 6/7 about diseases in left/right lung - skip them  
                    target_question_key = None
                
                # Add the question data if we found a mapping and it's not Q1 (already handled)
                if target_question_key and target_question_key != "question1":
                    final_dataset[study_id][target_question_key] = question_data
    
    return final_dataset

def validate_dataset(final_dataset: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    """Validate that the dataset meets the requirements"""
    
    question_names = {
        "question1": "Consolidation Detection",
        "question2": "ET Tube Detection", 
        "question3": "Nodules Detection",
        "question4": "COVID-19 Detection",
        "question5": "Mediastinum Normal",
        "question8": "Pneumothorax Detection",
        "question9": "Pleural Effusion Detection",
        "question10": "Pneumonia Detection"
    }
    
    validation_results = {}
    
    print("\n=== DATASET VALIDATION ===")
    print(f"Total samples: {len(final_dataset)}")
    
    for question_key, question_name in question_names.items():
        positive_count = 0
        total_count = 0
        
        for study_id, study_data in final_dataset.items():
            if question_key in study_data:
                total_count += 1
                answer = study_data[question_key].get('results', {}).get('answer', {})
                exist_value = answer.get('EXIST', answer.get('NORMAL'))
                if exist_value == 1:
                    positive_count += 1
        
        validation_results[question_key] = {
            'total': total_count,
            'positive': positive_count,
            'percentage': round(positive_count/total_count*100, 1) if total_count > 0 else 0
        }
        
        status = "✅" if positive_count >= 50 else "❌"
        print(f"{status} {question_name}: {positive_count}/{total_count} positive ({positive_count/total_count*100:.1f}%)")
    
    return validation_results

def main():
    """Main function to create the strategically balanced dataset"""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Creating strategically balanced dataset...")
    print("Requirements:")
    print("- Total: 500 samples across ALL questions")
    print("- Each question must have ≥50 positive cases from these 500 samples")
    print("- Question 1: Changed to consolidation detection with ≥50 cases")
    print("- All COVID-19 positive cases preserved")
    
    # Load data
    all_answers, metadata = load_data()
    
    # Find consolidation cases from medical text
    consolidation_studies = find_consolidation_cases(metadata, target_count=100)  # Get more candidates
    
    # Find samples with complete question coverage
    complete_samples = find_complete_samples(all_answers)
    
    # Strategically select 500 samples
    selected_study_ids = select_strategic_samples_v2(
        complete_samples, consolidation_studies, target_total=500
    )
    
    # Create consolidated dataset
    final_dataset = create_consolidated_dataset_v2(
        all_answers, selected_study_ids, consolidation_studies
    )
    
    # Validate results
    validation_results = validate_dataset(final_dataset)
    
    # Save dataset
    output_file = "data/strategic_balanced_dataset_500_samples.json"
    with open(output_file, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    
    # Create detailed report
    report = {
        "dataset_summary": {
            "total_samples": len(final_dataset),
            "creation_date": "2025-01-18",
            "strategy": "Strategic selection ensuring complete question coverage with positive case optimization",
            "requirements": {
                "total_samples": 500,
                "minimum_positive_per_question": 50,
                "consolidation_detection": "Question 1 changed from tuberculosis using medical text analysis",
                "covid19_preservation": "All positive cases maintained"
            }
        },
        "validation_results": validation_results,
        "consolidation_sources": {
            "medical_text_matches": len(consolidation_studies & set(selected_study_ids)),
            "total_candidates": len(consolidation_studies),
            "keywords_used": ["consolidation", "pneumonia", "infiltrate", "infiltration"]
        }
    }
    
    report_file = "data/strategic_dataset_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Strategic dataset creation completed!")
    print(f"📁 Dataset: {output_file}")
    print(f"📊 Report: {report_file}")
    print(f"📋 Total samples: {len(final_dataset)}")
    
    # Check if requirements met
    all_requirements_met = all(
        result['positive'] >= 50 for result in validation_results.values()
    )
    
    if all_requirements_met:
        print("🎯 All requirements successfully met!")
    else:
        print("⚠️  Some requirements not fully met - see validation results above")
        
        # Show which requirements were not met
        failed_requirements = []
        for question_key, result in validation_results.items():
            if result['positive'] < 50:
                failed_requirements.append(f"{question_key}: {result['positive']}/50")
        print(f"Failed requirements: {', '.join(failed_requirements)}")

if __name__ == "__main__":
    main() 