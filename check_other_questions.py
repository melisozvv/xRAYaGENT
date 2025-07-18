import json
import os

def check_other_questions():
    """Check what questions 8, 9, and 10 are about"""
    
    json_file = "data/gpt4_correct_answers_all_500_samples_20250715_224243.json"
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get first study to examine question content
        first_study = next(iter(data.values()))
        
        print("QUESTION CONTENT ANALYSIS:")
        print("="*60)
        
        for question_key in sorted(first_study.keys()):
            if question_key.startswith('question'):
                query = first_study[question_key].get('query', 'No query found')
                print(f"\n{question_key.upper()}:")
                print(f"Query: {query[:200]}...")  # First 200 characters
        
        print("\n" + "="*60)

if __name__ == "__main__":
    check_other_questions() 