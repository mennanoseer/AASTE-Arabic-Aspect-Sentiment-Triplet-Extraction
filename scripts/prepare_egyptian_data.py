"""
Script to prepare Egyptian dialect data for ASTE training.
Converts JSON format to triplet text format and splits into train/test/dev sets.
Normalizes punctuation to match ASTE format (space before punctuation marks).

Usage:
    python prepare_egyptian_data.py --category health
    python prepare_egyptian_data.py --category fashion
    python prepare_egyptian_data.py --category electronics
    python prepare_egyptian_data.py --category combined (merges all into one dataset)
"""

import json
import random
import os
import re
import argparse
from pathlib import Path


def normalize_punctuation(text):
    """
    Normalize punctuation in Arabic text to match ASTE format.
    - Adds space before sentence-ending punctuation (. ! ?)
    - Ensures text ends with proper punctuation
    
    Args:
        text: The review text to normalize
        
    Returns:
        Normalized text with proper spacing before punctuation
    """
    # Remove any existing spaces before punctuation first
    text = re.sub(r'\s+([.!?،؛])', r'\1', text.strip())
    
    # Add space before sentence-ending punctuation if not present
    text = re.sub(r'([^\s])([.!?])$', r'\1 \2', text)
    
    # If no punctuation at end, add space and period
    if text and not re.search(r'[.!?]$', text):
        text = text + ' .'
    
    return text


def convert_json_to_triplet_format(json_file, output_file):
    """
    Convert Egyptian dialect JSON data to triplet text format.
    
    Args:
        json_file: Path to the input JSON file
        output_file: Path to the output triplet text file
    
    Returns:
        int: Number of samples processed
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    triplet_lines = []
    skipped = 0
    
    for item in data:
        review_text = item.get('review_text', '')
        triplets_str = item.get('triplets', '[]')
        
        # Skip if no review text or empty triplets
        if not review_text or triplets_str == '[]':
            skipped += 1
            continue
        
        # Normalize punctuation to match ASTE format
        review_text = normalize_punctuation(review_text)
        
        # Format: review_text####triplets
        line = f"{review_text}####{triplets_str}\n"
        triplet_lines.append(line)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(triplet_lines)
    
    print(f"Converted {len(triplet_lines)} samples (skipped {skipped})")
    return len(triplet_lines)


def split_data(input_file, output_dir, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train/dev/test sets.
    
    Args:
        input_file: Path to the input triplet text file
        output_dir: Directory to save the split files
        train_ratio: Ratio for training data (default: 0.7)
        dev_ratio: Ratio for development data (default: 0.15)
        test_ratio: Ratio for test data (default: 0.15)
        seed: Random seed for reproducibility
    """
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle data
    random.seed(seed)
    random.shuffle(lines)
    
    # Calculate split indices
    total = len(lines)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    # Split data
    train_data = lines[:train_end]
    dev_data = lines[train_end:dev_end]
    test_data = lines[dev_end:]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write split files
    with open(os.path.join(output_dir, 'train_triplets.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    with open(os.path.join(output_dir, 'dev_triplets.txt'), 'w', encoding='utf-8') as f:
        f.writelines(dev_data)
    
    with open(os.path.join(output_dir, 'test_triplets.txt'), 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    
    print(f"\nData split complete:")
    print(f"  Train: {len(train_data)} samples ({train_ratio*100}%)")
    print(f"  Dev:   {len(dev_data)} samples ({dev_ratio*100}%)")
    print(f"  Test:  {len(test_data)} samples ({test_ratio*100}%)")
    print(f"\nFiles saved to: {output_dir}")


def main():
    """Main function to prepare Egyptian dialect data."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Prepare Egyptian dialect data for ASTE training')
    parser.add_argument('--category', type=str, default='health',
                       choices=['health', 'fashion', 'electronics', 'combined'],
                       help='Category to process: health, fashion, electronics, or combined (merged)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio for training data (default: 0.7)')
    parser.add_argument('--dev_ratio', type=float, default=0.15,
                       help='Ratio for development data (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio for test data (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    # Handle combined mode - merge all categories into one dataset
    if args.category == 'combined':
        print("\n" + "=" * 60)
        print("Preparing Combined Egyptian Dialect Data")
        print("=" * 60)
        
        all_categories = ['health', 'fashion', 'electronics']
        combined_lines = []
        
        # Collect data from all categories
        for category in all_categories:
            json_file = base_dir / 'datasets' / 'egyptian_dialect' / category / f'{category}_merged.json'
            
            if not json_file.exists():
                print(f"\nWarning: {json_file} not found. Skipping {category}.")
                continue
            
            print(f"\nReading {category} data from: {json_file}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                skipped = 0
                for item in data:
                    review_text = item.get('review_text', '')
                    triplets_str = item.get('triplets', '[]')
                    
                    if not review_text or triplets_str == '[]':
                        skipped += 1
                        continue
                    
                    review_text = normalize_punctuation(review_text)
                    line = f"{review_text}####{triplets_str}\n"
                    combined_lines.append(line)
                
                print(f"  Added {len(data) - skipped} samples from {category} (skipped {skipped})")
                
            except Exception as e:
                print(f"  Error reading {category}: {e}")
        
        if not combined_lines:
            print("\nError: No data found to combine.")
            return
        
        print(f"\nTotal combined samples: {len(combined_lines)}")
        
        # Create output directory
        output_dir = base_dir / 'datasets' / 'egyptian_dialect' / 'combined'
        temp_file = output_dir / 'combined_all_triplets.txt'
        
        # Write combined data to temporary file
        os.makedirs(output_dir, exist_ok=True)
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.writelines(combined_lines)
        
        # Split combined data
        print("\nSplitting combined data into train/dev/test...")
        split_data(
            input_file=temp_file,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        # Clean up temporary file
        os.remove(temp_file)
        print("\n" + "=" * 60)
        print("Combined Egyptian dialect data preparation complete.")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        return
    
    # Process single category
    category = args.category
    print("\n" + "=" * 60)
    print(f"Preparing Egyptian Dialect {category.capitalize()} Data")
    print("=" * 60)
    
    # Define paths for this category
    json_file = base_dir / 'datasets' / 'egyptian_dialect' / category / f'{category}_merged.json'
    temp_file = base_dir / 'datasets' / 'egyptian_dialect' / category / f'{category}_all_triplets.txt'
    output_dir = base_dir / 'datasets' / 'egyptian_dialect' / category
    
    # Check if JSON file exists
    if not json_file.exists():
        print(f"\nError: {json_file} not found.")
        return
    
    print(f"\nInput file: {json_file}")
    
    # Step 1: Convert JSON to triplet format
    print("\nStep 1: Converting JSON to triplet format...")
    num_samples = convert_json_to_triplet_format(json_file, temp_file)
    
    # Step 2: Split into train/dev/test
    print("\nStep 2: Splitting data into train/dev/test...")
    split_data(
        input_file=temp_file,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Clean up temporary file
    os.remove(temp_file)
    print("\n" + "=" * 60)
    print(f"Egyptian dialect {category} data preparation complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
