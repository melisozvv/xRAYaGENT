import json
import os
from collections import defaultdict

def analyze_complete_exist_distribution():
    """Analyze the distribution of EXIST 0/1 values for all questions"""
    
    # Dictionary to store counts for each question type
    question_stats = defaultdict(lambda: {'0': 0, '1': 0, 'total': 0})
    
    # List of JSON files to analyze
    json_files = [
        "data/gpt4_correct_answers_test_10_samples_20250715_213459.json",
        "data/gpt4_correct_answers_all_500_samples_20250715_224243.json",
        "data/gpt4_correct_answers_filter_samples_20250718_104754.json"
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"\nAnalyzing {json_file}...")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Process each study
            for study_id, study_data in data.items():
                # Process each question in the study
                for question_key, question_data in study_data.items():
                    if question_key.startswith('question'):
                        query = question_data.get('query', '')
                        results = question_data.get('results', {})
                        answer = results.get('answer', {})
                        
                        # Determine question type based on query content
                        question_type = ""
                        if "tuberculosis" in query.lower():
                            question_type = "Question 1: Tuberculosis Detection"
                        elif "et tube" in query.lower() or "carina" in query.lower():
                            question_type = "Question 2: ET Tube Detection"
                        elif "nodule" in query.lower():
                            question_type = "Question 3: Nodules Detection"
                        elif "covid" in query.lower():
                            question_type = "Question 4: COVID-19 Detection"
                        elif "mediastinum" in query.lower():
                            question_type = "Question 5: Mediastinum Normal"
                        elif "left lung" in query.lower() and "disease" in query.lower():
                            question_type = "Question 6: Left Lung Diseases"
                        elif "right lung" in query.lower() and "disease" in query.lower():
                            question_type = "Question 7: Right Lung Diseases"
                        elif "pneumothorax" in query.lower():
                            question_type = "Question 8: Pneumothorax Detection"
                        elif "pleural effusion" in query.lower():
                            question_type = "Question 9: Pleural Effusion Detection"
                        elif "pneumonia" in query.lower():
                            question_type = "Question 10: Pneumonia Detection"
                        elif "hilar lymphadenopathy" in query.lower():
                            question_type = "Question 11: Hilar Lymphadenopathy Detection"
                        else:
                            question_type = f"Other: {question_key}"
                        
                        # Get the exist/normal value
                        exist_value = None
                        if 'EXIST' in answer:
                            exist_value = answer['EXIST']
                        elif 'NORMAL' in answer:
                            exist_value = answer['NORMAL']
                        
                        if exist_value is not None:
                            question_stats[question_type]['total'] += 1
                            if exist_value == 0:
                                question_stats[question_type]['0'] += 1
                            elif exist_value == 1:
                                question_stats[question_type]['1'] += 1
        else:
            print(f"File not found: {json_file}")
    
    # Print detailed results
    print("\n" + "="*80)
    print("COMPLETE DISTRIBUTION OF EXIST 0/1 VALUES FOR EACH QUESTION")
    print("="*80)
    
    for question_type, stats in sorted(question_stats.items()):
        print(f"\n{question_type}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  EXIST/NORMAL = 0: {stats['0']} ({stats['0']/stats['total']*100:.1f}%)")
        print(f"  EXIST/NORMAL = 1: {stats['1']} ({stats['1']/stats['total']*100:.1f}%)")
        print(f"  Ratio (0:1): {stats['0']}:{stats['1']}")
    
    # Create comprehensive summary table
    print("\n" + "="*100)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*100)
    print(f"{'Question':<45} {'Total':<8} {'0s':<8} {'1s':<8} {'% 0s':<8} {'% 1s':<8} {'Condition':<15}")
    print("-" * 100)
    
    # Define condition prevalence for each question
    condition_map = {
        "Question 1: Tuberculosis Detection": "Very Rare",
        "Question 2: ET Tube Detection": "Uncommon", 
        "Question 3: Nodules Detection": "Rare",
        "Question 4: COVID-19 Detection": "Very Rare",
        "Question 5: Mediastinum Normal": "Common",
        "Question 8: Pneumothorax Detection": "Very Rare",
        "Question 9: Pleural Effusion Detection": "Uncommon",
        "Question 10: Pneumonia Detection": "Uncommon"
    }
    
    for question_type, stats in sorted(question_stats.items()):
        if question_type.startswith("Question"):
            total = stats['total']
            zeros = stats['0']
            ones = stats['1']
            pct_zeros = zeros/total*100 if total > 0 else 0
            pct_ones = ones/total*100 if total > 0 else 0
            
            # Get condition prevalence
            condition = condition_map.get(question_type, "Unknown")
            
            print(f"{question_type:<45} {total:<8} {zeros:<8} {ones:<8} {pct_zeros:<8.1f} {pct_ones:<8.1f} {condition:<15}")
    
    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    # Sort by percentage of positive cases (1s)
    sorted_by_prevalence = sorted(
        [(q, s) for q, s in question_stats.items() if q.startswith("Question")],
        key=lambda x: x[1]['1']/x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )
    
    print("\nConditions ranked by prevalence (% of positive cases):")
    for i, (question, stats) in enumerate(sorted_by_prevalence, 1):
        pct_positive = stats['1']/stats['total']*100 if stats['total'] > 0 else 0
        condition_name = question.split(": ")[1] if ": " in question else question
        print(f"{i:2d}. {condition_name:<35} {pct_positive:6.1f}% ({stats['1']:4d}/{stats['total']:4d})")
    
    return question_stats

if __name__ == "__main__":
    analyze_complete_exist_distribution() 