import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm

# Read CSV file
csv_path = 'data/processed/papers.csv'

print("Reading papers.csv...")
# Since the file is large, read in chunks
chunks = []
total_rows = 0
year_counter = Counter()

# First pass: count total rows for progress bar
print("Scanning file size...")
total_lines = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1  # Exclude header

# Second pass: process data with progress bar
with tqdm(total=total_lines, desc="Processing", unit="rows") as pbar:
    for chunk in pd.read_csv(csv_path, usecols=['year'], chunksize=50000, encoding='utf-8'):
        total_rows += len(chunk)
        # Filter out invalid years: must be between 1900 and 2026
        valid_years = chunk['year'].dropna().astype(int)
        valid_years = valid_years[(valid_years >= 1900) & (valid_years <= 2025)]
        year_counter.update(valid_years.tolist())
        pbar.update(len(chunk))

print(f"\nTotal rows: {total_rows:,}")
print(f"Valid papers (1900-2025): {sum(year_counter.values()):,}")
print(f"Number of unique years: {len(year_counter)}")

# Sort by year
sorted_years = sorted(year_counter.items())

print("\nYear Distribution Statistics:")
print("=" * 60)
print(f"{'Year':<10} {'Paper Count':<20} {'Cumulative %':<15}")
print("=" * 60)

cumulative = 0
total = sum(year_counter.values())  # Use valid papers count
for year, count in sorted_years:
    cumulative += count
    percentage = (cumulative / total) * 100
    print(f"{int(year):<10} {count:<20,} {percentage:<15.2f}%")

# Generate visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Line plot
years = [int(y) for y, _ in sorted_years]
counts = [c for _, c in sorted_years]

ax1.plot(years, counts, marker='o', linewidth=2, markersize=4)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Papers', fontsize=12)
ax1.set_title('Time Distribution of Papers', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Cumulative percentage plot
cumulative_pct = []
cum = 0
for year, count in sorted_years:
    cum += count
    cumulative_pct.append((cum / total) * 100)

ax2.plot(years, cumulative_pct, marker='s', linewidth=2, markersize=4, color='orange')
ax2.axhline(y=80, color='r', linestyle='--', label='80% threshold', linewidth=2)
ax2.axhline(y=70, color='g', linestyle='--', label='70% threshold', linewidth=2)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax2.set_title('Cumulative Percentage Distribution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('time_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to time_distribution_analysis.png")

# Recommended split thresholds
print("\n" + "=" * 60)
print("Recommended Time Split Thresholds:")
print("=" * 60)

thresholds = [70, 75, 80, 85, 90]
for threshold in thresholds:
    for i, (year, _) in enumerate(sorted_years):
        if cumulative_pct[i] >= threshold:
            print(f"{threshold}% of data: up to year {int(year)}")
            break
