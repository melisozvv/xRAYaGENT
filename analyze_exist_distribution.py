import json
import os
from collections import defaultdict, Counter

def analyze_exist_distribution():
    """Analyze the distribution of EXIST 0/1 values for each question"""
    
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
    
    # Print results
    print("\n" + "="*80)
    print("DISTRIBUTION OF EXIST 0/1 VALUES FOR EACH QUESTION")
    print("="*80)
    
    for question_type, stats in sorted(question_stats.items()):
        print(f"\n{question_type}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  EXIST/NORMAL = 0: {stats['0']} ({stats['0']/stats['total']*100:.1f}%)")
        print(f"  EXIST/NORMAL = 1: {stats['1']} ({stats['1']/stats['total']*100:.1f}%)")
        print(f"  Ratio (0:1): {stats['0']}:{stats['1']}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Question Type':<35} {'Total':<8} {'0s':<8} {'1s':<8} {'% 0s':<8} {'% 1s':<8}")
    print("-" * 80)
    
    for question_type, stats in sorted(question_stats.items()):
        total = stats['total']
        zeros = stats['0']
        ones = stats['1']
        pct_zeros = zeros/total*100 if total > 0 else 0
        pct_ones = ones/total*100 if total > 0 else 0
        
        # Shorten question type for table
        short_type = question_type.replace("Question ", "Q").replace(" Detection", "").replace(" Normal", "")
        print(f"{short_type:<35} {total:<8} {zeros:<8} {ones:<8} {pct_zeros:<8.1f} {pct_ones:<8.1f}")
    
    return question_stats

if __name__ == "__main__":
    analyze_exist_distribution() 