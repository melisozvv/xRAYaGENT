import pandas as pd
import os

def filter_impressions_by_word_count(input_file, output_file, min_words=5):
    """
    Filter impressions to keep only those with at least min_words.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        min_words (int): Minimum number of words required in impression
    """
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} records")
    
    # Check if 'impression' column exists
    if 'Impression' not in df.columns:
        print("Error: 'Impression' column not found in the dataset")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Filter impressions by word count
    def has_sufficient_words(impression):
        if pd.isna(impression):
            return False
        words = str(impression).strip().split()
        return len(words) >= min_words
    
    # Apply filter
    initial_count = len(df)
    df_filtered = df[df['Impression'].apply(has_sufficient_words)]
    final_count = len(df_filtered)
    removed_count = initial_count - final_count
    
    print(f"Filtered dataset: {final_count} records")
    print(f"Removed: {removed_count} records ({removed_count/initial_count*100:.1f}%)")
    
    # Show some examples of removed impressions
    print("\nExamples of removed impressions (< 5 words):")
    df_removed = df[~df['Impression'].apply(has_sufficient_words)]
    for i, impression in enumerate(df_removed['Impression'].dropna().head(10)):
        word_count = len(str(impression).strip().split())
        print(f"  {i+1}. \"{impression}\" ({word_count} words)")
    
    # Save filtered dataset
    df_filtered.to_csv(output_file, index=False)
    print(f"\nFiltered dataset saved to: {output_file}")

if __name__ == "__main__":
    input_file = "./data/test_metadata.csv"
    output_file = "./data/filter_test_metadata.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        exit(1)
    
    # Filter impressions
    filter_impressions_by_word_count(input_file, output_file, min_words=6)
