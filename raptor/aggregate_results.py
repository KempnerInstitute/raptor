import pandas as pd
import json
import os
import glob

# Set base directory to current directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# If running as script, use the script's directory. If notebook, use current.

print(f"Searching in {base_dir}")

probe_dirs = glob.glob(os.path.join(base_dir, '*_probes'))
all_results = {}

for p_dir in sorted(probe_dirs):
    dataset_name = os.path.basename(p_dir).replace('_probes', '')
    jsonl_path = os.path.join(p_dir, 'results.jsonl')
    
    if not os.path.exists(jsonl_path):
        continue
        
    print(f"\nProcessing {dataset_name}...")
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("Empty file")
        continue

    df = pd.DataFrame(data)
    
    # Identify metric column
    metric_cols = [c for c in df.columns if c.startswith('best_val_')]
    if not metric_cols:
        print(f"No metric found starting with best_val_")
        continue
    metric_col = metric_cols[0]
    print(f"Metric: {metric_col}")
    
    # Aggregate
    # Calculate mean and std
    agg = df.groupby('variant')[metric_col].agg(['mean', 'std', 'count'])
    print(agg)
    
    # Store results
    dataset_res = {}
    for variant, row in agg.iterrows():
        dataset_res[variant] = {
            'mean': row['mean'],
            'std': row['std'],
            'count': int(row['count'])
        }
    all_results[dataset_name] = dataset_res

# Save to JSON
out_path = os.path.join(base_dir, 'aggregated_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved results to {out_path}")

# Print Table
print("\nSummary Table:")
print(f"{'Variant':<10} | {'ADE20k':<15} | {'ImageNet':<15} | {'NYUD':<15}")
print("-" * 65)

variants = sorted(list({v for d in all_results.values() for v in d.keys()}))
datasets = sorted(all_results.keys())

for variant in variants:
    row = [f"{variant:<10}"]
    for dataset in datasets:
        if dataset in all_results and variant in all_results[dataset]:
            res = all_results[dataset][variant]
            mean = res['mean']
            std = res.get('std')
            if pd.isna(std) or std is None:
                val = f"{mean:.4f}"
            else:
                val = f"{mean:.4f} ± {std:.4f}"
            row.append(f"{val:<15}")
        else:
            row.append(f"{'-':<15}")
    print(" | ".join(row))
