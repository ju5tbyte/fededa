"""A script to merge multiple JSON list files into a single output file."""

import argparse
import json
import os
import sys
from typing import List


def merge_json_files(input_paths: List[str], output_path: str) -> None:
    """Merges multiple JSON files containing lists into one combined list.

    Args:
        input_paths: A list of paths to the source JSON files.
        output_path: The destination path for the merged JSON file.
    """
    combined_data = []

    for path in input_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check if the root element is a list to maintain structure.
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            continue

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # indent=2 for readability, ensure_ascii=False for Unicode support.
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(
            f"Successfully merged {len(input_paths)} files into {output_path}"
        )
    except IOError as e:
        print(f"Error writing to {output_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON files into one."
    )

    # Use nargs='+' to accept one or more input files.
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="List of input JSON file paths separated by space.",
    )

    # Single output path.
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path for the merged output JSON file.",
    )

    args = parser.parse_args()
    merge_json_files(args.inputs, args.output)


if __name__ == "__main__":
    main()
