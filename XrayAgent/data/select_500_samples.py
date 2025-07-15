#!/usr/bin/env python3
"""
Select 500 X-ray samples from test_metadata.csv based on specific medical questions
"""

import pandas as pd
import re
import random
from collections import defaultdict
from typing import List, Dict
import json

# Set random seed for reproducibility
random.seed(42)

class XRayMetadataSampler:
    """
    Sample X-ray cases from metadata based on medical questions
    """
    
    def __init__(self, metadata_file: str = 'data/test_metadata.csv'):
        self.metadata_file = metadata_file
        self.df = None
        
        # Target questions and related keywords
        self.target_categories = {
            'tuberculosis': {
                'keywords': [
                    'tuberculosis', 'tb', 'mycobacterium', 'granuloma', 'cavitary',
                    'miliary', 'tuberculous', 'cavitation', 'apical scarring',
                    'upper lobe fibrosis', 'ghon', 'hilar lymphadenopathy'
                ],
                'target_count': 40,
                'description': 'Cases with tuberculosis or TB-related findings'
            },
            'et_tube': {
                'keywords': [
                    'endotracheal', 'et tube', 'ett', 'intubation', 'tube', 'carina',
                    'distance', 'positioned', 'placement', 'airway', 'ventilator',
                    'mechanical ventilation', 'support devices'
                ],
                'target_count': 50,
                'description': 'Cases with ET tubes, especially with distance measurements'
            },
            'nodules': {
                'keywords': [
                    'nodule', 'nodular', 'mass', 'lesion', 'opacity', 'spot',
                    'pulmonary nodule', 'lung mass', 'solitary', 'multiple nodules',
                    'calcified nodule', 'noncalcified', 'ground glass'
                ],
                'target_count': 60,
                'description': 'Cases with nodules or masses with location information'
            },
            'fractures': {
                'keywords': [
                    'fracture', 'break', 'broken', 'rib fracture', 'clavicle fracture',
                    'vertebral fracture', 'compression fracture', 'healing fracture',
                    'old fracture', 'acute fracture', 'chest wall', 'skeletal'
                ],
                'target_count': 40,
                'description': 'Cases with bone fractures'
            },
            'mediastinum': {
                'keywords': [
                    'mediastinal', 'mediastinum', 'hilar', 'lymphadenopathy',
                    'enlarged', 'widening', 'shift', 'deviation', 'mass effect',
                    'cardiomediastinal', 'heart size', 'cardiomegaly'
                ],
                'target_count': 50,
                'description': 'Cases with mediastinal abnormalities'
            },
            'pleural_effusion': {
                'keywords': [
                    'effusion', 'pleural effusion', 'fluid', 'pleural fluid',
                    'hydrothorax', 'bilateral effusion', 'unilateral effusion',
                    'costophrenic', 'blunting', 'layering'
                ],
                'target_count': 75,
                'description': 'Cases with pleural effusion'
            },
            'pneumonia': {
                'keywords': [
                    'pneumonia', 'infiltrate', 'consolidation', 'airspace disease',
                    'opacity', 'infection', 'inflammatory', 'bronchopneumonia',
                    'lobar pneumonia', 'patchy', 'bilateral pneumonia'
                ],
                'target_count': 100,
                'description': 'Cases with pneumonia or consolidation'
            },
            'other_abnormal': {
                'keywords': [
                    'atelectasis', 'pneumothorax', 'edema', 'emphysema',
                    'fibrosis', 'scarring', 'chronic', 'abnormal', 'disease'
                ],
                'target_count': 75,
                'description': 'Other lung diseases and abnormalities'
            },
            'normal': {
                'keywords': [
                    'normal', 'clear', 'no active', 'unremarkable', 'within normal limits',
                    'no acute', 'negative', 'no abnormality'
                ],
                'target_count': 10,
                'description': 'Normal cases for comparison'
            }
        }
        
    def load_data(self):
        """Load the metadata CSV file"""
        print(f"📂 Loading data from {self.metadata_file}...")
        self.df = pd.read_csv(self.metadata_file)
        print(f"✅ Loaded {len(self.df)} records")
        return self.df
    
    def categorize_record(self, record: pd.Series) -> List[str]:
        """Categorize a record based on findings and impression"""
        categories = []
        
        # Combine findings and impression for analysis
        text_to_analyze = ""
        if pd.notna(record['Findings']):
            text_to_analyze += record['Findings'].lower() + " "
        if pd.notna(record['Impression']):
            text_to_analyze += record['Impression'].lower() + " "
        if pd.notna(record['Indication']):
            text_to_analyze += record['Indication'].lower() + " "
        
        # Check each category
        for category, info in self.target_categories.items():
            for keyword in info['keywords']:
                if keyword.lower() in text_to_analyze:
                    categories.append(category)
                    break
        
        # If no specific category found but has medical content, categorize as other_abnormal
        if not categories and any(word in text_to_analyze for word in 
                                ['opacity', 'density', 'abnormal', 'finding', 'disease']):
            categories.append('other_abnormal')
        
        return categories if categories else ['normal']
    
    def analyze_dataset(self):
        """Analyze the dataset to see what's available"""
        print("\n🔍 Analyzing dataset for target pathologies...")
        
        category_records = defaultdict(list)
        
        for idx, record in self.df.iterrows():
            categories = self.categorize_record(record)
            for category in categories:
                category_records[category].append(idx)
        
        print("\n📊 Available samples by category:")
        for category, info in self.target_categories.items():
            available = len(category_records[category])
            target = info['target_count']
            print(f"  • {category}: {available} available, {target} needed - {info['description']}")
        
        return category_records
    
    def select_samples(self, category_records: Dict) -> List[int]:
        """Select 500 samples with target distribution"""
        selected_indices = []
        selection_log = {}
        
        print("\n🎯 Selecting samples...")
        
        # First, try to get target counts for each category
        for category, info in self.target_categories.items():
            available_indices = category_records[category]
            target_count = info['target_count']
            
            if len(available_indices) >= target_count:
                selected = random.sample(available_indices, target_count)
            else:
                selected = available_indices
            
            selected_indices.extend(selected)
            selection_log[category] = len(selected)
            print(f"  • {category}: selected {len(selected)}/{target_count}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in selected_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        
        selected_indices = unique_indices
        
        # If we need more samples to reach 500, add from available pool
        if len(selected_indices) < 500:
            remaining_needed = 500 - len(selected_indices)
            
            # Get all unused indices
            all_indices = set(range(len(self.df)))
            used_indices = set(selected_indices)
            unused_indices = list(all_indices - used_indices)
            
            if len(unused_indices) >= remaining_needed:
                additional = random.sample(unused_indices, remaining_needed)
                selected_indices.extend(additional)
                print(f"  • Added {len(additional)} additional samples to reach 500")
        
        # If we have more than 500, trim to exactly 500
        if len(selected_indices) > 500:
            selected_indices = selected_indices[:500]
            print(f"  • Trimmed to exactly 500 samples")
        
        print(f"\n✅ Final selection: {len(selected_indices)} samples")
        return selected_indices
    
    def create_sample_dataset(self, selected_indices: List[int]) -> pd.DataFrame:
        """Create the final sample dataset"""
        sample_df = self.df.iloc[selected_indices].copy()
        
        # Add category information
        categories_list = []
        for idx in selected_indices:
            record = self.df.iloc[idx]
            categories = self.categorize_record(record)
            categories_list.append('; '.join(categories))
        
        sample_df['categories'] = categories_list
        
        # Add boolean flags for easy filtering
        for category in self.target_categories.keys():
            column_name = f'has_{category}'
            sample_df[column_name] = sample_df['categories'].str.contains(category, case=False, na=False)
        
        return sample_df
    
    def save_results(self, sample_df: pd.DataFrame, output_file: str = 'selected_500_samples.csv'):
        """Save the selected samples"""
        sample_df.to_csv(output_file, index=False)
        print(f"💾 Saved {len(sample_df)} samples to {output_file}")
        
        # Create summary statistics
        summary = {}
        summary['total_samples'] = len(sample_df)
        summary['category_distribution'] = {}
        
        for category in self.target_categories.keys():
            count = sample_df[f'has_{category}'].sum()
            summary['category_distribution'][category] = int(count)
        
        # Save summary
        with open('sample_selection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_summary(self, sample_df: pd.DataFrame, summary: Dict):
        """Print selection summary"""
        print("\n📋 Selection Summary:")
        print("=" * 50)
        print(f"Total samples: {summary['total_samples']}")
        
        print("\n🎯 Target Questions Coverage:")
        questions = [
            ("has_tuberculosis", "Does this X-ray have tuberculosis?"),
            ("has_et_tube", "Does it have the ET Tube? What's the distance between the Carina and the ET Tube?"),
            ("has_nodules", "Is there nodules in this X-ray? Where is the nodule?"),
            ("has_fractures", "Is this X-ray contain any fractures?"),
            ("has_mediastinum", "Is the Mediastinum normal?"),
            ("has_pleural_effusion", "Can you locate the Pleural effusion?"),
            ("has_pneumonia", "Is there pneumonia? Can you locate the Pneumonia?"),
            ("has_other_abnormal", "What disease can you find in the left lung? Right lung?")
        ]
        
        for column, question in questions:
            count = summary['category_distribution'].get(column.replace('has_', ''), 0)
            print(f"  • {question}")
            print(f"    → {count} samples available")
        
        print(f"\n📊 Distribution:")
        total = summary['total_samples']
        for category, count in summary['category_distribution'].items():
            percentage = (count / total) * 100
            print(f"  • {category}: {count} samples ({percentage:.1f}%)")
        
        # Show some example findings
        print(f"\n🔍 Sample Findings Examples:")
        for category in ['tuberculosis', 'et_tube', 'nodules', 'fractures', 'pleural_effusion']:
            if f'has_{category}' in sample_df.columns:
                category_samples = sample_df[sample_df[f'has_{category}'] == True]
                if len(category_samples) > 0:
                    example = category_samples.iloc[0]
                    print(f"\n{category.upper()} Example:")
                    print(f"  ID: {example['id']}")
                    print(f"  Findings: {example['Findings'][:200]}...")
                    print(f"  Impression: {example['Impression'][:100]}...")
    
    def run_selection(self):
        """Run the complete selection process"""
        print("🔬 X-Ray Sample Selection from Test Metadata")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Analyze dataset
        category_records = self.analyze_dataset()
        
        # Select samples
        selected_indices = self.select_samples(category_records)
        
        # Create sample dataset
        sample_df = self.create_sample_dataset(selected_indices)
        
        # Save results
        summary = self.save_results(sample_df)
        
        # Print summary
        self.print_summary(sample_df, summary)
        
        return sample_df, summary

def main():
    """Main function"""
    sampler = XRayMetadataSampler()
    sample_df, summary = sampler.run_selection()
    
    print(f"\n🎉 Successfully selected 500 samples!")
    print(f"📁 Files created:")
    print(f"  • selected_500_samples.csv - The selected samples")
    print(f"  • sample_selection_summary.json - Summary statistics")
    
    return sample_df

if __name__ == "__main__":
    main() 