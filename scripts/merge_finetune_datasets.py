"""A script to merge multiple JSON list files and shuffle them with a seed."""

import argparse
import json
import random
import sys
from typing import List


def merge_and_shuffle_json(input_paths: List[str], 
                           output_path: str, 
                           seed: int) -> None:
    """Merges JSON files, shuffles with a seed, and saves to a file.

    Args:
        input_paths: A list of paths to the source JSON files.
        output_path: The destination path for the merged JSON file.
        seed: The random seed for shuffling reproducibility.
    """
    combined_data = []

    # Set the random seed for reproducibility.
    random.seed(seed)

    for path in input_paths:
        try:
            # Use absolute or relative paths seamlessly with open().
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            continue

    # Shuffle the combined list in-place.
    random.shuffle(combined_data)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully merged {len(combined_data)} items into {output_path}")
        print(f"Used random seed: {seed}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Merge and shuffle multiple JSON files with a fixed seed.")
    
    parser.add_argument(
        '-i', '--inputs', nargs='+', required=True,
        help='List of input JSON file paths (absolute or relative).')
    
    parser.add_argument(
        '-o', '--output', required=True,
        help='Path for the shuffled output JSON file.')

    # Added seed argument with a default value of 42.
    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='Random seed for shuffling (default: 42).')

    args = parser.parse_args()
    merge_and_shuffle_json(args.inputs, args.output, args.seed)


if __name__ == "__main__":
    main()