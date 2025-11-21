#!/usr/bin/env python3
"""
Filter large mbox files to extract only emails FROM specified sender
"""

import mailbox
import json
from pathlib import Path
import argparse

def is_sender_email(message, sender_identifiers):
    """Check if email is FROM the specified sender"""
    
    # Check From header
    from_header = message.get('From', '').lower()
    
    # Also check X-From and Sender headers (common in exports)
    x_from = message.get('X-From', '').lower()
    sender = message.get('Sender', '').lower()
    
    for identifier in sender_identifiers:
        if identifier.lower() in from_header or identifier.lower() in x_from or identifier.lower() in sender:
            return True
    
    return False

def process_mbox_for_sender(mbox_path, output_path, sender_identifiers, sender_name="user"):
    """Extract only sender's emails from mbox file"""
    
    print(f"Processing: {mbox_path}")
    print(f"Looking for emails FROM {sender_name}...")
    
    mbox = mailbox.mbox(str(mbox_path))
    training_data = []
    sender_count = 0
    total_count = 0
    
    for message in mbox:
        total_count += 1
        
        if is_sender_email(message, sender_identifiers):
            sender_count += 1
            
            try:
                # Get email body
                body = None
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == 'text/plain':
                            body = part.get_payload(decode=True)
                            if body:
                                body = body.decode('utf-8', errors='ignore')
                            break
                else:
                    body = message.get_payload(decode=True)
                    if body:
                        body = body.decode('utf-8', errors='ignore')
                
                if body:
                    # Clean the text
                    body = body.strip()
                    
                    # Remove quoted replies (lines starting with >)
                    lines = body.split('\n')
                    clean_lines = [l for l in lines if not l.strip().startswith('>')]
                    body = '\n'.join(clean_lines)
                    
                    # Create training examples from paragraphs
                    paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]
                    
                    for para in paragraphs:
                        if 50 < len(para) < 1000:  # Good length for training
                            training_data.append({
                                "instruction": "Write in the user's style",
                                "input": f"Email paragraph",
                                "output": para
                            })
                            
            except Exception as e:
                continue
        
        if total_count % 1000 == 0:
            print(f"  Processed {total_count} emails, found {sender_count} from sender, extracted {len(training_data)} examples")
    
    print(f"\nFinal results:")
    print(f"  Total emails: {total_count}")
    print(f"  Sender's emails: {sender_count}")
    print(f"  Training examples: {len(training_data)}")
    
    # Save filtered training data
    if training_data:
        output_file = Path(output_path) / "sender_filtered_train.jsonl"
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        print(f"  Saved to: {output_file}")
    
    return training_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input mbox file or directory")
    parser.add_argument("--output", default="data/processed/", help="Output directory")
    parser.add_argument("--sender-emails", required=True, help="Comma-separated list of sender email addresses")
    parser.add_argument("--sender-name", default="user", help="Name to use in output messages")
    
    args = parser.parse_args()
    
    # Parse sender identifiers
    sender_identifiers = [s.strip() for s in args.sender_emails.split(',')]
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    all_training_data = []
    
    if input_path.is_file():
        # Process single mbox
        data = process_mbox_for_sender(input_path, output_path, sender_identifiers, args.sender_name)
        all_training_data.extend(data)
    else:
        # Process all mbox files in directory
        for mbox_file in input_path.glob("*mbox*"):
            data = process_mbox_for_sender(mbox_file, output_path, sender_identifiers, args.sender_name)
            all_training_data.extend(data)
    
    print(f"\n✓ Total training examples from all files: {len(all_training_data)}")

if __name__ == "__main__":
    main()
