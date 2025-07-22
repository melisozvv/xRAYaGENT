#!/usr/bin/env python3
"""
Performance Metrics Calculator for X-ray Analysis Results
Compares model predictions with GPT-4 ground truth across multiple evaluation metrics.

Usage:
    python calculate_performance_metrics.py

This script calculates accuracy, precision, recall, and F1 scores for:
- Enhanced X-ray Agent results (output/xray_analysis_results.json)
- Google COVID-19 detection results (output_google/xray_analysis_results.json)  
- MedGemma VQA results (output_medgemma/medgemma/medgemma_analysis_results.json)

All compared against GPT-4 ground truth (data/gpt4_correct_answers_balanced_test_20250718_153121.json)
"""

import json
from collections import defaultdict

def calculate_binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics (accuracy, precision, recall, F1)"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def calculate_multiclass_metrics(y_true_sets, y_pred_sets):
    """Calculate metrics for multi-label disease classification using 14 binary labels"""
    # Define the 14 possible diseases
    diseases = [
        "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", 
        "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", 
        "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture", 
        "Lung Opacity", "Enlarged Cardiomediastinum"
    ]
    
    sample_accuracies = []
    sample_precisions = []
    sample_recalls = []
    sample_f1s = []
    
    for true_set, pred_set in zip(y_true_sets, y_pred_sets):
        
        # Convert sets to binary vectors
        y_true_binary = [1 if disease in true_set else 0 for disease in diseases]
        y_pred_binary = [1 if disease in pred_set else 0 for disease in diseases]
        
        # Calculate binary classification metrics for this sample
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        total_labels = len(diseases)
        
        # Sample-level metrics based on binary classification across 14 diseases
        sample_accuracy = (tp + tn) / total_labels if total_labels > 0 else 0.0
        
        # Precision and recall based on binary classification
        sample_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sample_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1: harmonic mean of precision and recall
        if sample_precision + sample_recall > 0:
            sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall)
        else:
            sample_f1 = 0.0
        
        # Store individual sample metrics
        sample_accuracies.append(sample_accuracy)
        sample_precisions.append(sample_precision)
        sample_recalls.append(sample_recall)
        sample_f1s.append(sample_f1)
    
    # Calculate average metrics across all samples
    avg_accuracy = sum(sample_accuracies) / len(sample_accuracies) if sample_accuracies else 0.0
    avg_precision = sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
    avg_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
    avg_f1 = sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0
    
    return avg_accuracy, avg_precision, avg_recall, avg_f1

def analyze_enhanced_agent_results():
    """Analyze the Enhanced X-ray Agent results (original implementation)"""
    print('\n' + '='*80)
    print('ENHANCED X-RAY AGENT - PERFORMANCE METRICS (500 SAMPLES)')
    print('='*80)
    
    # Load files
    with open('output/xray_analysis_results_0722.json', 'r') as f:
        results = json.load(f)

    with open('data/gpt4_correct_answers_balanced_test_20250718_153121.json', 'r') as f:
        ground_truth = json.load(f)

    # Helper function to parse answer data
    def parse_answer_data(answer_data):
        """Parse answer data, handling both JSON objects and JSON strings"""
        if isinstance(answer_data, str):
            try:
                # Try to parse JSON string
                parsed_data = json.loads(answer_data)
                return parsed_data
            except json.JSONDecodeError:
                # If parsing fails, return as is
                return answer_data
        return answer_data

    # Collect data for each question
    question_metrics = {}

    for question_num in range(1, 12):
        question_key = f'question{question_num}'
        
        y_true = []
        y_pred = []
        y_true_sets = []
        y_pred_sets = []
        distances_true = []
        distances_pred = []
        
        for study_id in results:
            if study_id in ground_truth and question_key in results[study_id] and question_key in ground_truth[study_id]:
                result_data = results[study_id][question_key].get('results', {})
                truth_data = ground_truth[study_id][question_key]['results']['answer']
                
                # Parse result_data in case it contains JSON strings (especially for question 4)
                parsed_result_data = parse_answer_data(result_data)
                
                # Handle cases where the answer is a JSON string (like question 4 copied from Google results)
                if isinstance(parsed_result_data, str):
                    try:
                        parsed_result_data = json.loads(parsed_result_data)
                    except json.JSONDecodeError:
                        pass
                
                # Handle different data formats - always convert lists to compatible format
                result_data_processed = parsed_result_data
                
                # If it's a list of detection results, convert to single dict format
                if isinstance(parsed_result_data, list) and len(parsed_result_data) > 0:
                    first_element = parsed_result_data[0]
                    
                    # If first element is a string, try to parse it as JSON
                    if isinstance(first_element, str):
                        try:
                            first_element = json.loads(first_element)
                        except json.JSONDecodeError:
                            pass
                    
                    # Check if this looks like detection results
                    if isinstance(first_element, dict) and 'EXIST' in first_element:
                        # Check if any detection has EXIST=1
                        has_detection = any(item.get('EXIST', 0) == 1 for item in parsed_result_data if isinstance(item, dict))
                        result_exist_value = 1 if has_detection else 0
                        
                        # Extract location from first detection if available
                        first_location = first_element.get('LOCATION', '') if isinstance(first_element, dict) else ''
                        
                        # Convert to compatible format
                        result_data_processed = {
                            'EXIST': result_exist_value,
                            'LOCATION': first_location
                        }
                    else:
                        # Use first element if not detection format
                        result_data_processed = first_element
                        
                # Handle general response format with 'answer' field
                elif isinstance(parsed_result_data, dict) and 'answer' in parsed_result_data:
                    answer_content = parsed_result_data['answer']
                    
                    # Try to parse the answer content as JSON
                    if isinstance(answer_content, str):
                        try:
                            answer_json = json.loads(answer_content)
                            result_data_processed = answer_json
                        except json.JSONDecodeError:
                            # If not JSON, keep original
                            result_data_processed = parsed_result_data
                    else:
                        result_data_processed = answer_content

                # Binary classification (EXIST, NORMAL)
                if 'EXIST' in truth_data and 'EXIST' in result_data_processed:
                    y_true.append(truth_data['EXIST'])
                    y_pred.append(result_data_processed['EXIST'])
                elif 'NORMAL' in truth_data and 'NORMAL' in result_data_processed:
                    y_true.append(truth_data['NORMAL'])
                    y_pred.append(result_data_processed['NORMAL'])
                # Multi-class (DISEASES)
                elif 'DISEASES' in truth_data and 'DISEASES' in result_data_processed:
                    true_diseases = set(truth_data['DISEASES']) if truth_data['DISEASES'] else set()
                    pred_diseases = set(result_data_processed['DISEASES']) if result_data_processed['DISEASES'] else set()
                    y_true_sets.append(true_diseases)
                    y_pred_sets.append(pred_diseases)
                # Distance comparison
                elif 'DISTANCE' in truth_data and 'DISTANCE' in result_data_processed:
                    distances_true.append(truth_data['DISTANCE'] if truth_data['DISTANCE'] is not None else 0)
                    distances_pred.append(result_data_processed['DISTANCE'] if result_data_processed['DISTANCE'] is not None else 0)
                else:
                    print(f"No matching field found for study {study_id} and question {question_num}")
            else:
                print(f"Study {study_id} not found in ground truth")
        # Calculate metrics based on data type
        if y_true and y_pred:
            accuracy, precision, recall, f1 = calculate_binary_metrics(y_true, y_pred)
        elif y_true_sets and y_pred_sets:
            accuracy, precision, recall, f1 = calculate_multiclass_metrics(y_true_sets, y_pred_sets)
        elif distances_true and distances_pred:
            # Convert distance comparison to binary classification
            # 1 if distance difference ≤ 1.5 cm, 0 if > 1.5 cm
            distance_binary_true = [1] * len(distances_true)  # All should be accurate
            distance_binary_pred = [1 if abs(t - p) <= 1.5 else 0 for t, p in zip(distances_true, distances_pred)]
            accuracy, precision, recall, f1 = calculate_binary_metrics(distance_binary_true, distance_binary_pred)
        else:
            accuracy = precision = recall = f1 = 0
        
        question_metrics[question_num] = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'samples': len(y_true) + len(y_true_sets) + len(distances_true)
        }

    # Print table
    print('Question     Accuracy (%)  Precision (%) Recall (%)    F1 Score (%) Samples')
    print('-'*80)

    for q_num in range(1, 12):
        metrics = question_metrics[q_num]
        print('Question {:<3} {:<11.1f} {:<13.1f} {:<11.1f} {:<11.1f} {:<7d}'.format(
            q_num, 
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1'],
            metrics['samples']
        ))

    # Calculate overall metrics
    valid_metrics = [m for m in question_metrics.values() if m['samples'] > 0]
    
    if valid_metrics:
        avg_accuracy = sum(m['accuracy'] for m in valid_metrics) / len(valid_metrics)
        avg_precision = sum(m['precision'] for m in valid_metrics) / len(valid_metrics)
        avg_recall = sum(m['recall'] for m in valid_metrics) / len(valid_metrics)
        avg_f1 = sum(m['f1'] for m in valid_metrics) / len(valid_metrics)
        total_samples = sum(m['samples'] for m in valid_metrics)

        print('-'*80)
        print('AVERAGE      {:<11.1f} {:<13.1f} {:<11.1f} {:<11.1f} {:<7d}'.format(
            avg_accuracy, avg_precision, avg_recall, avg_f1, total_samples
        ))
        
        return {
            'name': 'Enhanced Agent',
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision, 
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'total_samples': total_samples,
            'question_metrics': question_metrics
        }
    else:
        print('No comparable data found')
        return None

def analyze_model_results(filename, name, answer_key):
    """Generic function to analyze any model's results"""
    print(f'\n{"="*80}')
    print(f'{name} RESULTS - PERFORMANCE METRICS (500 SAMPLES)')
    print(f'{"="*80}')
    
    # Helper function to parse answer data
    def parse_answer_data(answer_data):
        """Parse answer data, handling both JSON objects and JSON strings"""
        if isinstance(answer_data, str):
            try:
                # Try to parse JSON string
                parsed_data = json.loads(answer_data)
                return parsed_data
            except json.JSONDecodeError:
                # If parsing fails, return as is
                return answer_data
        return answer_data
    
    # Load ground truth
    with open('data/gpt4_correct_answers_balanced_test_20250718_153121.json', 'r') as f:
        ground_truth = json.load(f)
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print('Question     Accuracy (%)  Precision (%) Recall (%)    F1 Score (%) Samples')
    print('-'*80)
    
    valid_questions = []
    question_metrics = {}
    
    for q_num in range(1, 12):
        question_key = f'question{q_num}'
        matches = 0
        total = 0
        tp = tn = fp = fn = 0
        
        for study_id in results:
            if study_id in ground_truth and question_key in results[study_id] and question_key in ground_truth[study_id]:
                try:
                    if answer_key == 'answer':
                        pred_data = results[study_id][question_key]['answer']
                    else:
                        pred_data = results[study_id][question_key]['results']
                    
                    # Parse pred_data in case it contains JSON strings
                    pred_data = parse_answer_data(pred_data)
                    
                    # Handle different data formats - always convert lists to compatible format
                    if isinstance(pred_data, list) and len(pred_data) > 0:
                        first_element = pred_data[0]
                        
                        # If first element is a string, try to parse it as JSON
                        if isinstance(first_element, str):
                            try:
                                first_element = json.loads(first_element)
                            except json.JSONDecodeError:
                                pass
                        
                        # Check if this looks like detection results
                        if isinstance(first_element, dict) and 'EXIST' in first_element:
                            # Check if any detection has EXIST=1
                            has_detection = any(item.get('EXIST', 0) == 1 for item in pred_data if isinstance(item, dict))
                            
                            # Extract location from first detection if available
                            first_location = first_element.get('LOCATION', '') if isinstance(first_element, dict) else ''
                            
                            # Convert to compatible format
                            pred_data = {
                                'EXIST': 1 if has_detection else 0,
                                'LOCATION': first_location
                            }
                        else:
                            # Use first element if not detection format
                            pred_data = first_element
                            
                    # Handle general response format with 'answer' field
                    elif isinstance(pred_data, dict) and 'answer' in pred_data:
                        answer_content = pred_data['answer']
                        
                        # Try to parse the answer content as JSON
                        if isinstance(answer_content, str):
                            try:
                                answer_json = json.loads(answer_content)
                                pred_data = answer_json
                            except json.JSONDecodeError:
                                # If not JSON, keep original
                                pass
                        else:
                            pred_data = answer_content
                    
                    truth_data = ground_truth[study_id][question_key]['results']['answer']
                    total += 1
                    
                    # Check EXIST field
                    if 'EXIST' in truth_data and 'EXIST' in pred_data:
                        truth_val = truth_data['EXIST']
                        pred_val = pred_data['EXIST']
                        
                        if truth_val == pred_val:
                            matches += 1
                            if truth_val == 1:
                                tp += 1
                            else:
                                tn += 1
                        else:
                            if pred_val == 1:
                                fp += 1
                            else:
                                fn += 1
                    
                    # Check NORMAL field
                    elif 'NORMAL' in truth_data and 'NORMAL' in pred_data:
                        if truth_data['NORMAL'] == pred_data['NORMAL']:
                            matches += 1
                    
                    # Check DISEASES field
                    elif 'DISEASES' in truth_data and 'DISEASES' in pred_data:
                        truth_diseases = set(truth_data['DISEASES']) if truth_data['DISEASES'] else set()
                        pred_diseases = set(pred_data['DISEASES']) if pred_data['DISEASES'] else set()
                        if truth_diseases == pred_diseases:
                            matches += 1
                    
                    # Check DISTANCE field
                    elif 'DISTANCE' in truth_data and 'DISTANCE' in pred_data:
                        truth_dist = truth_data['DISTANCE'] if truth_data['DISTANCE'] is not None else 0
                        pred_dist = pred_data['DISTANCE'] if pred_data['DISTANCE'] is not None else 0
                        # Distance is accurate if difference is ≤ 1.5 cm
                        if abs(truth_dist - pred_dist) <= 1.5:
                            matches += 1
                            tp += 1  # Correct distance measurement
                        else:
                            fn += 1  # Incorrect distance measurement
                except:
                    continue
        
        # Calculate metrics
        accuracy = (matches / total * 100) if total > 0 else 0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        question_metrics[q_num] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': total
        }
        
        if total > 0:
            valid_questions.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'samples': total
            })
        
        print('Question {:<3} {:<11.1f} {:<13.1f} {:<11.1f} {:<11.1f} {:<7d}'.format(
            q_num, accuracy, precision, recall, f1, total
        ))
    
    # Calculate averages for questions with data
    if valid_questions:
        avg_accuracy = sum(q['accuracy'] for q in valid_questions) / len(valid_questions)
        avg_precision = sum(q['precision'] for q in valid_questions) / len(valid_questions)
        avg_recall = sum(q['recall'] for q in valid_questions) / len(valid_questions)
        avg_f1 = sum(q['f1'] for q in valid_questions) / len(valid_questions)
        total_samples = sum(q['samples'] for q in valid_questions)
        
        print('-'*80)
        print('AVERAGE      {:<11.1f} {:<13.1f} {:<11.1f} {:<11.1f} {:<7d}'.format(
            avg_accuracy, avg_precision, avg_recall, avg_f1, total_samples
        ))
        
        return {
            'name': name,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'total_samples': total_samples,
            'question_metrics': question_metrics
        }
    else:
        print('-'*80)
        print('No comparable data found')
        return None

def print_comparison_summary(results_list):
    """Print a summary comparison of all models"""
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        print("\nNo valid results to compare")
        return
    
    print('\n' + '='*100)
    print('MODEL COMPARISON SUMMARY')
    print('='*100)
    print(f'{"Model":<20} {"Avg Accuracy":<12} {"Avg Precision":<14} {"Avg Recall":<12} {"Avg F1":<10} {"Samples":<8}')
    print('-'*100)
    
    for result in valid_results:
        print(f'{result["name"]:<20} {result["avg_accuracy"]:<11.1f}% {result["avg_precision"]:<13.1f}% {result["avg_recall"]:<11.1f}% {result["avg_f1"]:<9.1f}% {result["total_samples"]:<8d}')
    
    # Find best performing model for each metric
    best_accuracy = max(valid_results, key=lambda x: x['avg_accuracy'])
    best_precision = max(valid_results, key=lambda x: x['avg_precision'])
    best_recall = max(valid_results, key=lambda x: x['avg_recall'])
    best_f1 = max(valid_results, key=lambda x: x['avg_f1'])
    
    print('\n' + '-'*100)
    print('BEST PERFORMANCE BY METRIC:')
    print(f'• Accuracy:  {best_accuracy["name"]} ({best_accuracy["avg_accuracy"]:.1f}%)')
    print(f'• Precision: {best_precision["name"]} ({best_precision["avg_precision"]:.1f}%)')
    print(f'• Recall:    {best_recall["name"]} ({best_recall["avg_recall"]:.1f}%)')
    print(f'• F1 Score:  {best_f1["name"]} ({best_f1["avg_f1"]:.1f}%)')

def main():
    """Main function to run all analyses"""
    print("="*100)
    print("X-RAY ANALYSIS PERFORMANCE EVALUATION")
    print("Comparing Model Predictions vs GPT-4 Ground Truth (500 Samples)")
    print("="*100)
    
    results_summary = []
    
    # Analyze Enhanced X-ray Agent (original)
    try:
        enhanced_results = analyze_enhanced_agent_results()
        results_summary.append(enhanced_results)
    except FileNotFoundError as e:
        print(f"Enhanced Agent results not found: {e}")
    except Exception as e:
        print(f"Error analyzing Enhanced Agent results: {e}")
    
     # Analyze Google COVID-19 Detection
    try:
        google_results = analyze_model_results(
            'output_google/xray_analysis_results.json', 
            'GOOGLE COVID-19', 
            'results'
        )
        results_summary.append(google_results)
    except FileNotFoundError as e:
        print(f"Google results not found: {e}")
    except Exception as e:
        print(f"Error analyzing Google results: {e}")
    
    # Analyze MedGemma VQA
    try:
        medgemma_results = analyze_model_results(
            'output_medgemma/medgemma/medgemma_analysis_results_0722.json', 
            'MEDGEMMA VQA', 
            'answer'
        )
        results_summary.append(medgemma_results)
    except FileNotFoundError as e:
        print(f"MedGemma results not found: {e}")
    except Exception as e:
        print(f"Error analyzing MedGemma results: {e}")
    
    # Print comparison summary
    print_comparison_summary(results_summary)
    
    print(f'\n{"="*100}')
    print("Analysis complete! Results saved above.")
    print(f'{"="*100}')

if __name__ == "__main__":
    main() 