import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Load datasets
print("Loading datasets...")
df1 = pd.read_csv('balanced_combined_lotl_dataset.csv')
df2 = pd.read_csv('final_balanced_volttyphoon_dataset.csv')
df3 = pd.read_csv('advanced_lotl_alternative_syntax_dataset.csv')

print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")
print(f"Dataset 3 shape: {df3.shape}")

# Standardize column names
print("\nStandardizing column names...")

# Dataset 1: Command, Label
df1_clean = df1[['Command', 'Label']].copy()
df1_clean.columns = ['command', 'label']
df1_clean['source'] = 'combined_lotl'

# Dataset 2: command, label
df2_clean = df2[['command', 'label']].copy()
df2_clean['source'] = 'volttyphoon'

# Dataset 3: command, label, technique, ...
df3_clean = df3[['command', 'label']].copy()
df3_clean['source'] = 'alternative_syntax'

# Combine all datasets
print("Combining datasets...")
combined_df = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)

# Clean labels (ensure binary: 0 or 1)
combined_df['label'] = pd.to_numeric(combined_df['label'], errors='coerce')
combined_df = combined_df.dropna(subset=['label'])
combined_df['label'] = combined_df['label'].astype(int)

# Remove duplicates
combined_df = combined_df.drop_duplicates(subset=['command'], keep='first')

print(f"\nCombined dataset shape: {combined_df.shape}")
print(f"Label distribution:\n{combined_df['label'].value_counts()}")
print(f"Source distribution:\n{combined_df['source'].value_counts()}")

# Create LLM Benchmarking Dataset
print("\nCreating LLM benchmarking dataset...")

# Balance the dataset
benign = combined_df[combined_df['label'] == 0].sample(n=min(1000, len(combined_df[combined_df['label'] == 0])), random_state=42)
malicious = combined_df[combined_df['label'] == 1].sample(n=min(1000, len(combined_df[combined_df['label'] == 1])), random_state=42)

benchmark_df = pd.concat([benign, malicious], ignore_index=True)
benchmark_df = benchmark_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add metadata for LLM evaluation
benchmark_df['command_id'] = range(1, len(benchmark_df) + 1)

print(f"\nBenchmark dataset shape: {benchmark_df.shape}")
print(f"Label distribution in benchmark:\n{benchmark_df['label'].value_counts()}")

# Save combined dataset
combined_df.to_csv('integrated_lotl_dataset.csv', index=False)
print("\n✓ Saved: integrated_lotl_dataset.csv")

# Save benchmarking dataset
benchmark_df.to_csv('llm_benchmarking_dataset.csv', index=False)
print("✓ Saved: llm_benchmarking_dataset.csv")

# Create evaluation format for LLMs (with prompt template)
evaluation_samples = []
for idx, row in benchmark_df.iterrows():
    evaluation_samples.append({
        "id": row['command_id'],
        "command": row['command'],
        "ground_truth": "LOTL/Malicious" if row['label'] == 1 else "Benign",
        "source": row['source'],
        "prompt": f"Analyze the following Windows command and determine if it represents a Living-off-the-Land (LOTL) attack or benign activity:\n\nCommand: {row['command']}\n\nClassify as: LOTL/Malicious or Benign"
    })

# Save as JSON for easy LLM integration
with open('llm_evaluation_prompts.json', 'w') as f:
    json.dump(evaluation_samples, f, indent=2)
print("✓ Saved: llm_evaluation_prompts.json")

# Create summary report
summary = {
    "total_samples": len(benchmark_df),
    "benign_count": len(benchmark_df[benchmark_df['label'] == 0]),
    "malicious_count": len(benchmark_df[benchmark_df['label'] == 1]),
    "sources": benchmark_df['source'].value_counts().to_dict(),
    "balance_ratio": f"{len(benchmark_df[benchmark_df['label'] == 0]) / len(benchmark_df) * 100:.1f}% benign, {len(benchmark_df[benchmark_df['label'] == 1]) / len(benchmark_df) * 100:.1f}% malicious"
}

with open('benchmark_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*50)
print("BENCHMARK DATASET SUMMARY")
print("="*50)
print(f"Total samples: {summary['total_samples']}")
print(f"Benign: {summary['benign_count']}")
print(f"Malicious: {summary['malicious_count']}")
print(f"Balance: {summary['balance_ratio']}")
print(f"\nSources distribution:")
for source, count in summary['sources'].items():
    print(f"  - {source}: {count}")
print("="*50)
