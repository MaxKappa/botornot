#!/usr/bin/env python3
"""
Preprocess raw tweet data files
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import TweetPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw tweet JSON files'
    )
    parser.add_argument(
        '--input-dir',
        default='./data',
        help='Input directory with raw tweet files'
    )
    parser.add_argument(
        '--output-dir',
        default='./data_processed',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--num-files',
        type=int,
        default=8,
        help='Number of tweet files to process (default: 8)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TWEET PREPROCESSING")
    print("="*60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of files:  {args.num_files}")
    print()
    
    # Create preprocessor
    preprocessor = TweetPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process all files
    preprocessor.process_all_files(num_files=args.num_files)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
