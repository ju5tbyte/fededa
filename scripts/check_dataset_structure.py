"""Script to check MMCircuitEval dataset structure.

This script loads a sample from the dataset and prints its structure
to verify the format of explanations and other fields.
"""

from datasets import load_dataset

# Load dataset
data = load_dataset("charlie314159/MMCircuitEval", split="general")

print("=" * 80)
print("DATASET STRUCTURE ANALYSIS")
print("=" * 80)
print(f"Total samples: {len(data)}")
print()

# Check first sample
sample = data[0]
print("Keys in first sample:")
for key in sample.keys():
    print(f"  - {key}")
print()

# Check types and values
print("Field details for first sample:")
print(f"  answers: type={type(sample['answers'])}, len={len(sample['answers'])}")
print(f"  answers[0]: type={type(sample['answers'][0])}, value={repr(sample['answers'][0])}")
print()

if "explanations" in sample:
    print(f"  explanations: type={type(sample['explanations'])}, len={len(sample['explanations'])}")
    print(f"  explanations[0]: type={type(sample['explanations'][0])}, value={repr(sample['explanations'][0])}")
else:
    print("  explanations: NOT FOUND in dataset")
print()

print(f"  questions: type={type(sample['questions'])}, len={len(sample['questions'])}")
print(f"  question_types: type={type(sample['question_types'])}, len={len(sample['question_types'])}")
print(f"  abilities: type={type(sample['abilities'])}, len={len(sample['abilities'])}")
print(f"  images: type={type(sample['images'])}, len={len(sample['images'])}")
print()

# Check if explanations field exists and what it contains
if "explanations" in sample:
    print("Checking all samples for explanation patterns:")
    empty_count = 0
    none_count = 0
    non_empty_count = 0

    for i in range(min(100, len(data))):  # Check first 100 samples
        explanations = data[i]["explanations"]
        for exp in explanations:
            if exp is None:
                none_count += 1
            elif exp == "":
                empty_count += 1
            else:
                non_empty_count += 1

    print(f"  None values: {none_count}")
    print(f"  Empty strings: {empty_count}")
    print(f"  Non-empty: {non_empty_count}")
    print()

    # Show some examples
    print("Example non-empty explanation (if any):")
    for i in range(min(100, len(data))):
        explanations = data[i]["explanations"]
        for exp in explanations:
            if exp and exp != "":
                print(f"  Sample {i}: {repr(exp[:100])}")
                break
        else:
            continue
        break
else:
    print("'explanations' field does not exist in dataset!")

print("=" * 80)
