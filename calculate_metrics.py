#!/usr/bin/env python3
"""
Calculate accuracy, recall, and precision for EXIST field comparison
between AI agent results and GPT-4 ground truth.
"""

import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_exist_values(data, question_num, is_agent_result=True):
    """Extract EXIST values for a specific question from the data"""
    exist_values = {}
    
    for study_id, study_data in data.items():
        question_key = f"question{question_num}"
        if question_key in study_data:
            try:
                if is_agent_result:
                    # AI agent results: data[study_id][question_key]['results']['EXIST']
                    exist_value = study_data[question_key]['results']['EXIST']
                else:
                    # GPT-4 ground truth: data[study_id][question_key]['results']['answer']['EXIST']
                    exist_value = study_data[question_key]['results']['answer']['EXIST']
                
                exist_values[study_id] = exist_value
            except (KeyError, TypeError):
                # Skip if EXIST field doesn't exist or is malformed
                continue
    
    return exist_values

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and other metrics"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_samples': 0
        }
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_samples': len(y_true)
    }

def get_question_name(question_num):
    """Get descriptive question names"""
    question_names = {
        1: "Tuberculosis",
        2: "ET Tube",
        3: "Nodules",
        4: "COVID-19",
        5: "Mediastinum Normal",
        8: "Pneumothorax",
        9: "Pleural Effusion",
        10: "Pneumonia",
        11: "Hilar Lymphadenopathy"
    }
    return question_names.get(question_num, f"Question {question_num}")

def main():
    print("🔍 Loading data files...")
    
    # Load both JSON files
    agent_results = load_json_data('output/xray_analysis_results.json')
    ground_truth = load_json_data('data/gpt4_correct_answers_all_500_samples_20250715_224243.json')
    
    print(f"✅ Loaded {len(agent_results)} agent results")
    print(f"✅ Loaded {len(ground_truth)} ground truth records")
    
    # Questions to analyze (excluding 6 and 7 as they don't have EXIST field)
    questions_to_analyze = [1, 2, 3, 4, 5, 8, 9, 10, 11]
    
    results_table = []
    
    print("\n📊 Calculating metrics for each question...")
    
    for question_num in questions_to_analyze:
        print(f"\n🔄 Processing Question {question_num}: {get_question_name(question_num)}")
        
        # Extract EXIST values for this question
        agent_exist = extract_exist_values(agent_results, question_num, is_agent_result=True)
        truth_exist = extract_exist_values(ground_truth, question_num, is_agent_result=False)
        
        # Find common study IDs (intersection)
        common_study_ids = set(agent_exist.keys()) & set(truth_exist.keys())
        
        if len(common_study_ids) == 0:
            print(f"⚠️  No common study IDs found for question {question_num}")
            continue
        
        # Create aligned arrays for comparison
        y_true = []  # Ground truth
        y_pred = []  # Agent predictions
        
        for study_id in common_study_ids:
            y_true.append(truth_exist[study_id])
            y_pred.append(agent_exist[study_id])
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Count ground truth positives (EXIST=1)
        ground_truth_positives = sum(y_true)
        
        # Add to results table
        results_table.append({
            'Question': f"Q{question_num}",
            'Question_Name': get_question_name(question_num),
            'Total_Samples': metrics['total_samples'],
            'Ground_Truth_Positives': ground_truth_positives,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1_Score': f"{metrics['f1_score']:.3f}",
            'True_Positives': metrics['true_positives'],
            'False_Positives': metrics['false_positives'],
            'True_Negatives': metrics['true_negatives'],
            'False_Negatives': metrics['false_negatives']
        })
        
        print(f"   Samples: {metrics['total_samples']}")
        print(f"   Ground Truth Positives: {ground_truth_positives}")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
    
    # Create DataFrame and display results
    df = pd.DataFrame(results_table)
    
    print("\n" + "="*100)
    print("📋 FINAL RESULTS TABLE - ACCURACY, PRECISION, RECALL BY QUESTION")
    print("="*100)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('question_metrics_comparison.csv', index=False)
    print(f"\n💾 Results saved to 'question_metrics_comparison.csv'")
    
    # Calculate overall averages
    print("\n" + "="*50)
    print("📊 OVERALL AVERAGES:")
    print("="*50)
    
    # Convert string values back to float for averaging
    accuracies = [float(row['Accuracy']) for row in results_table]
    precisions = [float(row['Precision']) for row in results_table]
    recalls = [float(row['Recall']) for row in results_table]
    f1_scores = [float(row['F1_Score']) for row in results_table]
    
    print(f"Average Accuracy:  {np.mean(accuracies):.3f}")
    print(f"Average Precision: {np.mean(precisions):.3f}")
    print(f"Average Recall:    {np.mean(recalls):.3f}")
    print(f"Average F1-Score:  {np.mean(f1_scores):.3f}")
    
    total_samples = sum(row['Total_Samples'] for row in results_table)
    print(f"Total Samples Analyzed: {total_samples}")
    
    return df

if __name__ == "__main__":
    results_df = main() 