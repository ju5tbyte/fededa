
import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union
import argparse
import random
from tqdm import tqdm

# Define the paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "circuit_understanding"

class BaseDatasetProcessor(ABC):
    """
    Abstract base class for dataset processors.
    To add a new dataset, inherit from this class and implement the `process` method.
    """
    def __init__(self, raw_data_path: Path, output_dir: Path):
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = "dataset" # Override this in subclasses

    @abstractmethod
    def process(self) -> List[Dict[str, Any]]:
        """
        Process the dataset and return a list of entries in the target format.
        """
        pass

    def save_data(self, data: List[Dict[str, Any]], filename: str = None):
        if filename is None:
            filename = f"{self.dataset_name}_finetune.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(data)} entries to {output_path}")

    def copy_image(self, src_path: Path, new_filename: str) -> str:
        """
        Copies an image to the output images directory and returns the relative path.
        """
        dst_path = self.images_dir / new_filename
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        
        # Return path relative to the output directory (e.g., "images/filename.jpg")
        return f"images/{new_filename}"


class ElectroVizQAProcessor(BaseDatasetProcessor):
    """
    Processor for the ElectroVizQA dataset.
    """
    def __init__(self, raw_data_path: Path, output_dir: Path):
        super().__init__(raw_data_path, output_dir)
        self.dataset_name = "electrovizqa"
        self.image_source_dir = self.raw_data_path.parent / "full_images"

    def process(self) -> List[Dict[str, Any]]:
        print(f"Processing {self.dataset_name} from {self.raw_data_path}...")
        
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed_data = []
        
        # ElectroVizQA structure: {"1": {...}, "2": {...}}
        for question_id, entry in tqdm(raw_data.items(), desc=f"Processing {self.dataset_name}"):
            try:
                # 1. Image handling
                image_filename = entry.get("image")
                if not image_filename:
                    continue
                
                # Fix: Some entries have trailing spaces in the filename
                image_filename = image_filename.strip()
                
                src_image_path = self.image_source_dir / image_filename
                if not src_image_path.exists():
                    print(f"Warning: Image not found {src_image_path}")
                    continue
                
                # Create a unique filename to avoid collisions with other datasets
                new_image_filename = f"electroviz_{image_filename}"
                relative_image_path = self.copy_image(src_image_path, new_image_filename)

                # 2. Construct Conversation
                # The 'Hint' field usually contains the formatted question + choices.
                # If 'Hint' is missing, fallback to formatting it manually.
                user_content = entry.get("Hint")
                if not user_content:
                    question = entry.get("question", "")
                    choices = entry.get("choices", [])
                    formatted_choices = " ".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                    user_content = f"Please answer the question and provide the correct option letter, e.g., A, B, C, D at the end.\nQuestion: {question}\nChoices: {formatted_choices}"

                # Ensure <image> token is present
                if "<image>" not in user_content:
                    user_content = "<image>\n" + user_content
                
                gpt_content = entry.get("correct_option") # e.g., "C"

                if not user_content or not gpt_content:
                    continue

                # 3. Create Entry
                conversation_entry = {
                    "image": relative_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": user_content
                        },
                        {
                            "from": "gpt",
                            "value": gpt_content
                        }
                    ]
                }
                
                processed_data.append(conversation_entry)
            
            except Exception as e:
                print(f"Error processing entry {question_id}: {e}")
                continue

        return processed_data


class CircuitVQAProcessor(BaseDatasetProcessor):
    """
    Processor for the CircuitVQA dataset.
    """
    def __init__(self, raw_data_path: Path, output_dir: Path, target_prefixes: List[str] = None, sampling_ratios: Dict[str, float] = None, seed: int = 42):
        super().__init__(raw_data_path, output_dir)
        self.dataset_name = "circuitvqa"
        self.target_prefixes = target_prefixes
        self.sampling_ratios = sampling_ratios or {}
        self.seed = seed
        # raw_data_path is .../question_answer/master.json
        # Images are in .../images/model-inputs/
        # Structure: data/raw/CircuitVQA/images/model-inputs/{splittype}/{file}.jpg
        self.image_root = self.raw_data_path.parent.parent / "images" / "model-inputs"

    def process(self) -> List[Dict[str, Any]]:
        print(f"Processing {self.dataset_name} from {self.raw_data_path}...")
        
        try:
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            print(f"Error loading raw data: {e}")
            return []

        processed_data = {}
        
        # CircuitVQA structure is a list of dicts
        for entry in tqdm(raw_data, desc=f"Processing {self.dataset_name}"):
            try:
                split_type = entry.get("splittype") # e.g., 'train', 'val', 'test'
                file_stem = entry.get("file")
                
                # Filter by prefixes if specified (e.g., 'd4' for logic circuits)
                if self.target_prefixes:
                    if not any(file_stem.startswith(prefix) for prefix in self.target_prefixes):
                        continue
                
                if not split_type or not file_stem:
                    continue
                
                # Images are typically .jpg in this dataset structure
                image_filename = f"{file_stem}.jpg"
                src_image_path = self.image_root / split_type / image_filename
                
                if not src_image_path.exists():
                    # Check if file exists without extension or with other extensions if needed
                    # But based on analysis, .jpg is correct.
                    # print(f"Warning: Image not found {src_image_path}")
                    continue
                
                # Create unique filename
                new_image_filename = f"circuitvqa_{image_filename}"
                relative_image_path = self.copy_image(src_image_path, new_image_filename)

                # Construct Conversation
                question = entry.get("question")
                answer = entry.get("answer")
                
                if not question or answer is None:
                    continue
                
                user_content = f"<image>\n{question}"
                gpt_content = str(answer) # Answer might be integer (count)

                # Create Entry
                conversation_entry = {
                    "image": relative_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": user_content
                        },
                        {
                            "from": "gpt",
                            "value": gpt_content
                        }
                    ]
                }
                
                if split_type not in processed_data:
                    processed_data[split_type] = []
                processed_data[split_type].append(conversation_entry)
            
            except Exception as e:
                # print(f"Error processing entry: {e}")
                continue

        # Apply sampling if configured
        if self.sampling_ratios:
            random.seed(self.seed)
            print(f"Applying sampling with seed {self.seed}...")
            for split, items in processed_data.items():
                ratio = self.sampling_ratios.get(split, 1.0)
                if ratio < 1.0:
                    original_count = len(items)
                    sample_size = int(original_count * ratio)
                    # Use sample to get unique elements if items are unique, 
                    # but here items are dicts which are not hashable, so random.sample works on the list index or directly on list
                    processed_data[split] = random.sample(items, sample_size)
                    print(f"  - {split}: Sampled {sample_size}/{original_count} ({ratio:.1%})")

        return processed_data


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for circuit understanding fine-tuning.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save processed data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    
    # --- Register Processors Here ---
    # You can add more processors to this list as you add more datasets
    processors = [
        ElectroVizQAProcessor(
            raw_data_path=PROJECT_ROOT / "external/ElectroVizQA/data/full_data.json", 
            output_dir=output_dir
        ),
        CircuitVQAProcessor(
            raw_data_path=PROJECT_ROOT / "data/raw/CircuitVQA/question_answer/master.json",
            output_dir=output_dir,
            target_prefixes=["d4"], # Filter for logic circuits as requested
            sampling_ratios={"train": 1.0, "val": 1.0, "test": 1.0}, # Example: Set to 0.1 for 10% data
            seed=42
        )
    ]
    # --------------------------------


    
    for processor in processors:
        if processor.raw_data_path.exists():
            data = processor.process()
            
            if isinstance(data, dict):
                # Handle split datasets (e.g. {'train': [...], 'val': [...]})
                for split_name, split_data in data.items():
                    processor.save_data(split_data, filename=f"{processor.dataset_name}_{split_name}_finetune.json")
            else:
                # Handle single list datasets
                processor.save_data(data, filename=f"{processor.dataset_name}_finetune.json")
        else:
            print(f"Skipping {processor.dataset_name}: Source file not found at {processor.raw_data_path}")



if __name__ == "__main__":
    main()
