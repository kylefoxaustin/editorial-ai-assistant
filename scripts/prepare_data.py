#!/usr/bin/env python3
"""
Data Preparation Script for Editorial AI Assistant
Converts raw editorial examples into training format
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd

class EditorialDataPrep:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def create_training_example(self,
                               original_text: str,
                               edited_text: str,
                               edit_type: str = "general",
                               explanation: str = None) -> Dict:
        """Create a single training example"""

        # Build the instruction based on edit type
        instructions = {
            "general": "Edit this text for clarity and style",
            "grammar": "Fix grammar and punctuation errors",
            "clarity": "Improve clarity and readability",
            "concision": "Make this text more concise",
            "tone": "Adjust the tone to be more professional",
            "development": "Provide developmental editing feedback"
        }

        instruction = instructions.get(edit_type, instructions["general"])

        # Format for fine-tuning
        example = {
            "instruction": instruction,
            "input": original_text,
            "output": edited_text
        }

        if explanation:
            example["output"] += f"\n\n[Editorial Notes: {explanation}]"

        return example

    def process_csv(self, csv_path: str) -> List[Dict]:
        """Process a CSV file with original/edited pairs"""
        df = pd.read_csv(csv_path)
        examples = []

        for _, row in df.iterrows():
            example = self.create_training_example(
                original_text=row['original'],
                edited_text=row['edited'],
                edit_type=row.get('type', 'general'),
                explanation=row.get('explanation', None)
            )
            examples.append(example)

        return examples

    def save_dataset(self, examples: List[Dict], output_name: str = "training_data"):
        """Save processed examples in multiple formats"""

        # Save as JSONL (for fine-tuning)
        jsonl_path = self.processed_dir / f"{output_name}.jsonl"
        with open(jsonl_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        print(f"✓ Saved {len(examples)} examples to {jsonl_path}")

        # Save as JSON (for inspection)
        json_path = self.processed_dir / f"{output_name}.json"
        with open(json_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"✓ Saved readable version to {json_path}")

        # Create train/validation split
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        # Save splits
        with open(self.processed_dir / "train.jsonl", 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')

        with open(self.processed_dir / "validation.jsonl", 'w') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')

        print(f"✓ Created train/val split: {len(train_examples)}/{len(val_examples)}")

        return jsonl_path

def main():
    """Process editorial data from CSV"""
    prep = EditorialDataPrep()

    # Check if we have a CSV file
    csv_file = Path("data/raw/sample_editorial_data.csv")

    if csv_file.exists():
        print(f"Processing CSV file: {csv_file}")
        examples = prep.process_csv(csv_file)
        print(f"Loaded {len(examples)} examples from CSV")
    else:
        # Create a few sample examples if no CSV
        print("No CSV found, creating sample examples...")
        examples = [
            prep.create_training_example(
                "The report which was written by John was very comprehensive and it contained alot of good informations.",
                "John's report was comprehensive and contained extensive valuable information.",
                edit_type="concision",
                explanation="Reduced wordiness, fixed 'alot' spelling, corrected 'informations' to 'information'"
            ),
            prep.create_training_example(
                "We need to leverage our core competencies to synergize our value proposition.",
                "We should focus on our strengths to improve our customer offering.",
                edit_type="clarity",
                explanation="Removed corporate jargon for clearer communication"
            ),
            prep.create_training_example(
                "The defendant was found to be not guilty by the jury.",
                "The jury found the defendant not guilty.",
                edit_type="concision",
                explanation="Active voice is more direct and concise"
            )
        ]

    # Save the dataset
    prep.save_dataset(examples, "editorial_training_data")

    print("\n📝 Dataset Statistics:")
    print(f"  Total examples: {len(examples)}")

    # Count types
    from collections import Counter
    types = Counter([ex.get('instruction', '').split()[0] for ex in examples])
    for edit_type, count in types.most_common():
        print(f"  {edit_type}: {count}")

    print("\n✅ Ready for training!")
    print("Next step: python scripts/train_model.py --use-8bit")

if __name__ == "__main__":
    main()
