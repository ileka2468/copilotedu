from transformers import PreTrainedTokenizerFast
from pathlib import Path
import sys

# --- CONFIGURATION ---
# Make sure these paths are correct for your project structure
TOKENIZER_PATH = "dataset/tokenizer.json"
TRAIN_EQUATIONS_PATH = "dataset/train/equations.txt"
VAL_EQUATIONS_PATH = "dataset/val/equations.txt"
# ---------------------

def scan_file(file_path: Path, tokenizer, vocab_size: int):
    print(f"\nScanning {file_path.name}...")
    found_issue = False
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Tokenize the line just like the training script does
                    token_ids = tokenizer.encode(line)
                    
                    # Check if any token ID is out of bounds
                    for token_id in token_ids:
                        if token_id >= vocab_size:
                            print(f"❌ Found invalid token ID!")
                            print(f"   - File: {file_path.name}")
                            print(f"   - Line Number: {i + 1}")
                            print(f"   - Invalid Token ID: {token_id} (Vocab size is {vocab_size})")
                            print(f"   - Line Text: '{line}'")
                            found_issue = True
                            
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}")
                    print(f"   - Line Text: '{line}'")
                    found_issue = True

    except FileNotFoundError:
        print(f"⚠️  File not found: {file_path}. Please update the path in the script.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

    if not found_issue:
        print(f"✅ No issues found in {file_path.name}.")
        
    return found_issue


if __name__ == "__main__":
    # Update the paths in the CONFIGURATION section above before running!
    if "path/to/your" in TRAIN_EQUATIONS_PATH:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE EDIT THE SCRIPT AND UPDATE THE PATHS TO YOUR DATA !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
        vocab_size = len(tokenizer)
        print(f"Tokenizer loaded. Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer from {TOKENIZER_PATH}. Error: {e}")
        sys.exit(1)

    scan_file(Path(TRAIN_EQUATIONS_PATH), tokenizer, vocab_size)
    scan_file(Path(VAL_EQUATIONS_PATH), tokenizer, vocab_size)