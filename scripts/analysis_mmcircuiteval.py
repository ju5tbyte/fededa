"""Analysis script for MMCircuitEval dataset distribution.

This script loads the MMCircuitEval dataset from Hugging Face and analyzes
the distribution of various metadata fields across different splits.
"""

from collections import Counter
from typing import Optional

from datasets import load_dataset


def analyze_field_distribution(
    dataset_name: str = "charlie314159/MMCircuitEval",
    fields: Optional[list[str]] = None
) -> dict[str, dict[str, Counter]]:
    """Analyze the distribution of specified fields across all splits.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        fields (Optional[list[str]]): List of fields to analyze. If None,
            defaults to common metadata fields.

    Returns:
        dict[str, dict[str, Counter]]: Nested dictionary where first key is
            split name, second key is field name, and value is Counter object
            with distribution.
    """
    if fields is None:
        fields = [
            "question_types",
            "difficulties",
            "abilities",
            "ic_type",
            "source",
            "extra"
        ]

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    print(f"\nAvailable splits: {list(dataset.keys())}")
    print(f"Fields to analyze: {fields}\n")

    results = {}

    for split_name in dataset.keys():
        print(f"Analyzing split: {split_name}")
        split_data = dataset[split_name]
        print(f"  Total samples: {len(split_data)}")

        results[split_name] = {}

        for field in fields:
            if field not in split_data.column_names:
                print(f"  Warning: Field '{field}' not found in split '{split_name}'")
                continue

            # Get all values for this field
            values = split_data[field]

            # Handle list fields (like 'abilities' might be a list)
            flat_values = []
            for val in values:
                if isinstance(val, list):
                    flat_values.extend(val)
                else:
                    flat_values.append(val)

            # Count occurrences
            counter = Counter(flat_values)
            results[split_name][field] = counter

            print(f"  Field '{field}': {len(counter)} unique values")

    return results


def print_distribution_report(results: dict[str, dict[str, Counter]]) -> None:
    """Print a detailed distribution report.

    Args:
        results (dict[str, dict[str, Counter]]): Results from analyze_field_distribution.
    """
    print("\n" + "=" * 80)
    print("DISTRIBUTION REPORT")
    print("=" * 80)

    for split_name, split_results in results.items():
        print(f"\n{'=' * 80}")
        print(f"SPLIT: {split_name.upper()}")
        print(f"{'=' * 80}")

        for field_name, counter in split_results.items():
            print(f"\n{'-' * 80}")
            print(f"Field: {field_name}")
            print(f"{'-' * 80}")

            # Sort by count (descending)
            sorted_items = counter.most_common()

            total_count = sum(counter.values())
            print(f"Total occurrences: {total_count}")
            print(f"Unique values: {len(counter)}\n")

            # Print distribution
            for value, count in sorted_items:
                percentage = (count / total_count) * 100
                print(f"  {value}: {count} ({percentage:.2f}%)")


def save_distribution_to_file(
    results: dict[str, dict[str, Counter]],
    output_path: str = "outputs/mmcircuiteval_distribution.txt"
) -> None:
    """Save distribution report to a text file.

    Args:
        results (dict[str, dict[str, Counter]]): Results from analyze_field_distribution.
        output_path (str): Path where the report will be saved.
    """
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MMCircuitEval Dataset Distribution Report\n")
        f.write("=" * 80 + "\n")

        for split_name, split_results in results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SPLIT: {split_name.upper()}\n")
            f.write(f"{'=' * 80}\n")

            for field_name, counter in split_results.items():
                f.write(f"\n{'-' * 80}\n")
                f.write(f"Field: {field_name}\n")
                f.write(f"{'-' * 80}\n")

                sorted_items = counter.most_common()
                total_count = sum(counter.values())
                f.write(f"Total occurrences: {total_count}\n")
                f.write(f"Unique values: {len(counter)}\n\n")

                for value, count in sorted_items:
                    percentage = (count / total_count) * 100
                    f.write(f"  {value}: {count} ({percentage:.2f}%)\n")

    print(f"\nDistribution report saved to: {output_path}")


def main() -> None:
    """Main function to run the analysis."""
    # Analyze the dataset
    results = analyze_field_distribution()

    # Print the report
    print_distribution_report(results)

    # Save to file
    save_distribution_to_file(results)


if __name__ == "__main__":
    main()
