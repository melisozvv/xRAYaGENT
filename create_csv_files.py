#!/usr/bin/env python3
"""
CSV File Generator for X-ray Analysis Performance Metrics
Creates CSV files from the updated performance metrics results.

Updated metrics include:
- 1.5cm threshold for distance measurements
- 18-disease binary classification for multi-label tasks
- Improved evaluation criteria

Generates:
- enhanced_agent_metrics.csv
- google_covid19_metrics.csv  
- medgemma_vqa_metrics.csv
- model_comparison_summary.csv
"""

import csv

def get_question_description(question_num):
    """Get description for each question"""
    descriptions = {
        1: "Consolidation Detection",
        2: "ET Tube & Carina Distance", 
        3: "Nodule Detection",
        4: "COVID-19 Detection",
        5: "Mediastinum Assessment",
        6: "Left Lung Disease Identification",
        7: "Right Lung Disease Identification", 
        8: "Pneumothorax Detection",
        9: "Pleural Effusion Location",
        10: "Pneumonia Detection",
        11: "Hilar Lymphadenopathy Detection"
    }
    return descriptions.get(question_num, f"Question {question_num}")

def create_enhanced_agent_csv():
    """Create CSV file for Enhanced X-ray Agent results"""
    data = [
        {"Question": 1, "Task": get_question_description(1), "Accuracy": 67.0, "Precision": 20.0, "Recall": 29.9, "F1_Score": 24.0, "Samples": 443},
        {"Question": 2, "Task": get_question_description(2), "Accuracy": 89.7, "Precision": 59.3, "Recall": 56.1, "F1_Score": 57.7, "Samples": 456},
        {"Question": 3, "Task": get_question_description(3), "Accuracy": 58.2, "Precision": 14.1, "Recall": 44.8, "F1_Score": 21.5, "Samples": 455},
        {"Question": 4, "Task": get_question_description(4), "Accuracy": 23.4, "Precision": 8.2, "Recall": 100.0, "F1_Score": 15.1, "Samples": 440},
        {"Question": 5, "Task": get_question_description(5), "Accuracy": 56.4, "Precision": 49.5, "Recall": 52.3, "F1_Score": 50.9, "Samples": 456},
        {"Question": 6, "Task": get_question_description(6), "Accuracy": 60.5, "Precision": 9.0, "Recall": 33.3, "F1_Score": 11.4, "Samples": 456},
        {"Question": 7, "Task": get_question_description(7), "Accuracy": 60.8, "Precision": 9.3, "Recall": 32.4, "F1_Score": 11.7, "Samples": 456},
        {"Question": 8, "Task": get_question_description(8), "Accuracy": 59.0, "Precision": 11.7, "Recall": 42.9, "F1_Score": 18.3, "Samples": 456},
        {"Question": 9, "Task": get_question_description(9), "Accuracy": 50.9, "Precision": 20.7, "Recall": 67.5, "F1_Score": 31.7, "Samples": 456},
        {"Question": 10, "Task": get_question_description(10), "Accuracy": 56.4, "Precision": 27.8, "Recall": 58.3, "F1_Score": 37.6, "Samples": 456},
        {"Question": 11, "Task": get_question_description(11), "Accuracy": 59.4, "Precision": 6.2, "Recall": 35.5, "F1_Score": 10.6, "Samples": 456}
    ]
    
    # Add average row
    data.append({
        "Question": "AVERAGE", "Task": "Overall Performance", 
        "Accuracy": 58.3, "Precision": 21.4, "Recall": 50.3, "F1_Score": 26.4, "Samples": 4986
    })
    
    return data

def create_google_covid19_csv():
    """Create CSV file for Google COVID-19 results"""
    data = [
        {"Question": 1, "Task": get_question_description(1), "Accuracy": 29.6, "Precision": 19.7, "Recall": 91.9, "F1_Score": 32.5, "Samples": 500},
        {"Question": 2, "Task": get_question_description(2), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 3, "Task": get_question_description(3), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 4, "Task": get_question_description(4), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 5, "Task": get_question_description(5), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 6, "Task": get_question_description(6), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 7, "Task": get_question_description(7), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 8, "Task": get_question_description(8), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 9, "Task": get_question_description(9), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 10, "Task": get_question_description(10), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0},
        {"Question": 11, "Task": get_question_description(11), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 0}
    ]
    
    # Add average row (only for Question 1 since others are 0)
    data.append({
        "Question": "AVERAGE", "Task": "Overall Performance", 
        "Accuracy": 29.6, "Precision": 19.7, "Recall": 91.9, "F1_Score": 32.5, "Samples": 500
    })
    
    return data

def create_medgemma_vqa_csv():
    """Create CSV file for MedGemma VQA results"""
    data = [
        {"Question": 1, "Task": get_question_description(1), "Accuracy": 78.0, "Precision": 44.6, "Recall": 52.9, "F1_Score": 48.4, "Samples": 500},
        {"Question": 2, "Task": get_question_description(2), "Accuracy": 13.0, "Precision": 12.7, "Recall": 100.0, "F1_Score": 22.5, "Samples": 500},
        {"Question": 3, "Task": get_question_description(3), "Accuracy": 22.0, "Precision": 13.6, "Recall": 95.3, "F1_Score": 23.8, "Samples": 500},
        {"Question": 4, "Task": get_question_description(4), "Accuracy": 60.2, "Precision": 11.6, "Recall": 75.8, "F1_Score": 20.2, "Samples": 500},
        {"Question": 5, "Task": get_question_description(5), "Accuracy": 45.6, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 500},
        {"Question": 6, "Task": get_question_description(6), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 500},
        {"Question": 7, "Task": get_question_description(7), "Accuracy": 0.2, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0, "Samples": 500},
        {"Question": 8, "Task": get_question_description(8), "Accuracy": 89.6, "Precision": 53.3, "Recall": 15.1, "F1_Score": 23.5, "Samples": 500},
        {"Question": 9, "Task": get_question_description(9), "Accuracy": 25.8, "Precision": 18.3, "Recall": 100.0, "F1_Score": 30.9, "Samples": 500},
        {"Question": 10, "Task": get_question_description(10), "Accuracy": 48.8, "Precision": 28.2, "Recall": 88.9, "F1_Score": 42.9, "Samples": 500},
        {"Question": 11, "Task": get_question_description(11), "Accuracy": 58.0, "Precision": 8.1, "Recall": 48.6, "F1_Score": 13.9, "Samples": 500}
    ]
    
    # Add average row
    data.append({
        "Question": "AVERAGE", "Task": "Overall Performance", 
        "Accuracy": 40.1, "Precision": 17.3, "Recall": 52.4, "F1_Score": 20.6, "Samples": 5500
    })
    
    return data

def create_model_comparison_csv():
    """Create CSV file for model comparison summary"""
    data = [
        {"Model": "Enhanced X-ray Agent", "Avg_Accuracy": 58.3, "Avg_Precision": 21.4, "Avg_Recall": 50.3, "Avg_F1": 26.4, "Total_Samples": 4986},
        {"Model": "Google COVID-19", "Avg_Accuracy": 29.6, "Avg_Precision": 19.7, "Avg_Recall": 91.9, "Avg_F1": 32.5, "Total_Samples": 500},
        {"Model": "MedGemma VQA", "Avg_Accuracy": 40.1, "Avg_Precision": 17.3, "Avg_Recall": 52.4, "Avg_F1": 20.6, "Total_Samples": 5500}
    ]
    
    return data

def save_to_csv(data, filename, fieldnames):
    """Save data to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"✓ Saved {filename}")

def print_summary_table(data, title):
    """Print a formatted summary table"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Question':<10} {'Task':<30} {'Acc%':<6} {'Prec%':<7} {'Rec%':<6} {'F1%':<6} {'Samples':<8}")
    print('-'*80)
    
    for row in data:
        if row['Question'] == 'AVERAGE':
            print('-'*80)
            print(f"{'AVG':<10} {row['Task'][:29]:<30} {row['Accuracy']:<5.1f} {row['Precision']:<6.1f} {row['Recall']:<5.1f} {row['F1_Score']:<5.1f} {row['Samples']:<8}")
        else:
            print(f"Q{row['Question']:<9} {row['Task'][:29]:<30} {row['Accuracy']:<5.1f} {row['Precision']:<6.1f} {row['Recall']:<5.1f} {row['F1_Score']:<5.1f} {row['Samples']:<8}")

def main():
    """Main function to generate all CSV files"""
    print("="*100)
    print("GENERATING UPDATED PERFORMANCE METRICS CSV FILES")
    print("="*100)
    print("Updates Applied:")
    print("• Distance threshold: 1.5cm (improved from 0.5cm)")
    print("• Multi-label evaluation: 18-disease binary classification")
    print("• Comprehensive binary classification metrics")
    print("="*100)
    
    # Generate Enhanced X-ray Agent CSV
    enhanced_data = create_enhanced_agent_csv()
    fieldnames_questions = ['Question', 'Task', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'Samples']
    save_to_csv(enhanced_data, 'enhanced_agent_metrics.csv', fieldnames_questions)
    print_summary_table(enhanced_data, "ENHANCED X-RAY AGENT METRICS")
    
    # Generate Google COVID-19 CSV
    google_data = create_google_covid19_csv()
    save_to_csv(google_data, 'google_covid19_metrics.csv', fieldnames_questions)
    print_summary_table(google_data, "GOOGLE COVID-19 DETECTION METRICS")
    
    # Generate MedGemma VQA CSV
    medgemma_data = create_medgemma_vqa_csv()
    save_to_csv(medgemma_data, 'medgemma_vqa_metrics.csv', fieldnames_questions)
    print_summary_table(medgemma_data, "MEDGEMMA VQA METRICS")
    
    # Generate Model Comparison CSV
    comparison_data = create_model_comparison_csv()
    fieldnames_comparison = ['Model', 'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_F1', 'Total_Samples']
    save_to_csv(comparison_data, 'model_comparison_summary.csv', fieldnames_comparison)
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Avg Acc%':<9} {'Avg Prec%':<10} {'Avg Rec%':<9} {'Avg F1%':<8} {'Samples':<8}")
    print('-'*80)
    for row in comparison_data:
        print(f"{row['Model']:<20} {row['Avg_Accuracy']:<8.1f} {row['Avg_Precision']:<9.1f} {row['Avg_Recall']:<8.1f} {row['Avg_F1']:<7.1f} {row['Total_Samples']:<8}")
    
    print(f"\n{'='*100}")
    print("CSV FILES SUCCESSFULLY GENERATED:")
    print("• enhanced_agent_metrics.csv - Detailed metrics for Enhanced X-ray Agent")
    print("• google_covid19_metrics.csv - Detailed metrics for Google COVID-19 detection")
    print("• medgemma_vqa_metrics.csv - Detailed metrics for MedGemma VQA")
    print("• model_comparison_summary.csv - Overall model comparison")
    print(f"{'='*100}")
    
    # Create analysis summary
    print("\nKEY FINDINGS:")
    print("• Best Overall Accuracy: Enhanced X-ray Agent (58.3%)")
    print("• Best Overall Precision: Enhanced X-ray Agent (21.4%)")
    print("• Best Overall Recall: Google COVID-19 (91.9%)")
    print("• Best Overall F1 Score: Google COVID-19 (32.5%)")
    print("• Best Individual Task: Enhanced Agent Q2 - ET Tube Distance (89.7% accuracy)")
    print("• Most Challenging Tasks: Disease identification (Questions 6-7)")

if __name__ == "__main__":
    main() 