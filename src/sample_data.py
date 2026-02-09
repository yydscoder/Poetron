"""
Sample a smaller subset of the poetry dataset for faster training
"""
import pandas as pd
from pathlib import Path


def create_sample_dataset(input_file, output_file, sample_fraction=0.23, random_state=42):
    """
    Create a smaller sample of the dataset for faster training.
    
    Args:
        input_file (str): Path to the full dataset CSV
        output_file (str): Path to save the sampled dataset
        sample_fraction (float): Fraction of data to sample (0-1)
        random_state (int): Random seed for reproducibility
    
    Returns:
        int: Number of rows in the sampled dataset
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original dataset size: {len(df)} rows")
    
    # Sample the data
    print(f"Sampling {sample_fraction*100}% of the data...")
    sampled_df = df.sample(frac=sample_fraction, random_state=random_state)
    print(f"Sampled dataset size: {len(sampled_df)} rows")
    
    # Save the smaller dataset
    sampled_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    return len(sampled_df)


if __name__ == '__main__':
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'PoetryFoundationData.csv'
    output_file = data_dir / 'PoetryFoundationData_small.csv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        exit(1)
    
    create_sample_dataset(str(input_file), str(output_file), sample_fraction=0.01)
    print("\nDone! You can now use the smaller dataset for training.")
