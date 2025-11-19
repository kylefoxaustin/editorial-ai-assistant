#!/usr/bin/env python3
"""
Enhanced Data Preparation for Editorial AI
Handles multiple input formats
"""

import json
import pandas as pd
from pathlib import Path
import argparse

def process_csv(file_path):
    """Process CSV with columns: original, edited, type, explanation"""
    df = pd.read_csv(file_path)
    examples = []

    for _, row in df.iterrows():
        example = {
            "instruction": f"Edit for {row.get('type', 'general improvement')}",
            "input": row['original'],
            "output": row['edited']
        }
        if 'explanation' in row and pd.notna(row['explanation']):
            example['explanation'] = row['explanation']
        examples.append(example)

    return examples

def process_docx(file_path):
    """Process Word document with tracked changes as training data"""
    try:
        from docx import Document
        doc = Document(file_path)

        print("Processing Word document...")
        print("Note: This expects alternating paragraphs of original/edited text")

        examples = []
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for i in range(0, len(paragraphs)-1, 2):
            example = {
                "instruction": "Edit for clarity and style",
                "input": paragraphs[i],
                "output": paragraphs[i+1] if i+1 < len(paragraphs) else paragraphs[i]
            }
            examples.append(example)

        return examples
    except ImportError:
        print("python-docx not installed. Run: pip install python-docx")
        return []

def process_xlsx(file_path):
    """Process Excel file with editorial data"""
    df = pd.read_excel(file_path)
    # Same structure as CSV
    examples = []

    for _, row in df.iterrows():
        example = {
            "instruction": f"Edit for {row.get('type', 'general improvement')}",
            "input": row['original'],
            "output": row['edited']
        }
        if 'explanation' in row and pd.notna(row['explanation']):
            example['explanation'] = row['explanation']
        examples.append(example)

    return examples

def show_format_examples():
    """Show examples of expected data formats"""
    print("""
EXPECTED DATA FORMATS:

1. CSV Format:
----------------
original,edited,type,explanation
"The report which was written...","The report written...","concision","Removed unnecessary words"
"Its important to note...","It's important to note...","grammar","Fixed apostrophe"

2. JSONL Format (one JSON object per line):
----------------
{"instruction": "Fix grammar", "input": "Original text here", "output": "Edited text here"}
{"instruction": "Make concise", "input": "Another original", "output": "Another edited"}

3. Word Document (.docx):
----------------
Paragraph 1: Original text that needs editing
Paragraph 2: The edited version of that text
Paragraph 3: Another original text
Paragraph 4: Its edited version

4. Excel (.xlsx):
----------------
| original | edited | type | explanation |
|----------|--------|------|-------------|
| Text 1   | Edit 1 | grammar | Fixed errors |
| Text 2   | Edit 2 | concision | Made shorter |
""")

def main():
    parser = argparse.ArgumentParser(description="Prepare editorial training data")
    parser.add_argument('--input', default='data/raw/sample_editorial_data.csv',
                       help='Input file path')
    parser.add_argument('--input-type', choices=['csv', 'jsonl', 'docx', 'xlsx', 'auto'],
                       default='auto', help='Input file type')
    parser.add_argument('--output', default='data/processed/',
                       help='Output directory for processed data')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show format examples and exit')

    args = parser.parse_args()

    if args.show_examples:
        show_format_examples()
        return

    # Auto-detect file type
    file_path = Path(args.input)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    if args.input_type == 'auto':
        suffix = file_path.suffix.lower()
        type_map = {
            '.csv': 'csv',
            '.jsonl': 'jsonl',
            '.docx': 'docx',
            '.xlsx': 'xlsx',
            '.xls': 'xlsx'
        }
        args.input_type = type_map.get(suffix, 'csv')
        print(f"Auto-detected file type: {args.input_type}")

    # Process based on type
    print(f"Processing {file_path}...")

    if args.input_type == 'csv':
        examples = process_csv(file_path)
    elif args.input_type == 'docx':
        examples = process_docx(file_path)
    elif args.input_type == 'xlsx':
        examples = process_xlsx(file_path)
    elif args.input_type == 'jsonl':
        # Process JSONL
        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
    else:
        print(f"Unsupported type: {args.input_type}")
        return

    if not examples:
        print("No examples found in input file!")
        return

    # Save processed data
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Split train/val (90/10)
    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val = examples[split_idx:] if split_idx < len(examples) else [examples[-1]]

    # Save
    with open(output_dir / 'train.jsonl', 'w') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')

    with open(output_dir / 'validation.jsonl', 'w') as f:
        for ex in val:
            f.write(json.dumps(ex) + '\n')

    print(f"✓ Processed {len(examples)} examples successfully!")
    print(f"  Training examples: {len(train)}")
    print(f"  Validation examples: {len(val)}")
    print(f"  Output saved to: {output_dir}")
    print(f"\nNext step: Run training with these processed files")

if __name__ == "__main__":
    main()
