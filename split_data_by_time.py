import json
from tqdm import tqdm

# Threshold year: 80% of data is before 2021
YEAR_THRESHOLD = 2021
BATCH_SIZE = 100000  # Process records in batches

input_file = 'data/raw/output.jsonl'
train_file = 'data/raw/train.jsonl'
test_file = 'data/raw/test.jsonl'

print(f"Splitting data with threshold year: {YEAR_THRESHOLD}")
print(f"Train set: papers from 1900 to {YEAR_THRESHOLD} (≈80%)")
print(f"Test set: papers from {YEAR_THRESHOLD + 1} onwards (≈20%)")

# Count total lines first for progress bar
print("\nScanning JSONL file size...")
total_lines = sum(1 for _ in open(input_file, encoding='utf-8'))

train_count = 0
test_count = 0
skipped_count = 0

print(f"\nProcessing {total_lines:,} records...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(train_file, 'w', encoding='utf-8') as train_out, \
     open(test_file, 'w', encoding='utf-8') as test_out:
    
    train_batch = []
    test_batch = []
    
    for line_num, line in tqdm(enumerate(infile, 1), total=total_lines, desc="Splitting", unit="records"):
        try:
            record = json.loads(line)
            year = record.get('year')
            
            # Filter valid years (1900-2026)
            if year is None or year < 1900 or year > 2025:
                skipped_count += 1
            elif year <= YEAR_THRESHOLD:
                train_batch.append(line)
                train_count += 1
            else:
                test_batch.append(line)
                test_count += 1
            
            # Write batches when size reached
            if len(train_batch) >= BATCH_SIZE:
                train_out.writelines(train_batch)
                train_batch = []
            if len(test_batch) >= BATCH_SIZE:
                test_out.writelines(test_batch)
                test_batch = []
                
        except json.JSONDecodeError:
            skipped_count += 1
            continue
    
    # Write remaining records
    if train_batch:
        train_out.writelines(train_batch)
    if test_batch:
        test_out.writelines(test_batch)

print(f"\n" + "=" * 60)
print(f"Train set: {train_count:,} records ({train_count/(train_count+test_count)*100:.2f}%)")
print(f"Test set: {test_count:,} records ({test_count/(train_count+test_count)*100:.2f}%)")
print(f"Skipped (invalid years): {skipped_count:,} records")
print(f"Total processed: {train_count + test_count + skipped_count:,}")
print("=" * 60)

print(f"\nTrain data saved to: {train_file}")
print(f"Test data saved to: {test_file}")
