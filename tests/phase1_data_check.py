"""
phase1_data_check.py
--------------------
Phase 1: data check for the learning-to-rank experiment.
Reads models/weak_label_training_data.csv, reports columns,
group stats, positive rate per group, and drops groups with no positives.
"""

import sys
import pandas as pd
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "models" / "weak_label_training_data.csv"

print("=" * 60)
print("PHASE 1: DATA CHECK")
print("=" * 60)

df = pd.read_csv(csv_path)

# 1. Columns
print(f"\nFile: {csv_path.name}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# 2. jd_id presence
if "jd_id" not in df.columns:
    print("\nERROR: jd_id column NOT found. Cannot proceed with group stats.")
    sys.exit(1)

print("\njd_id column FOUND. Proceeding with group stats.")

# Group stats
groups = df.groupby("jd_id")
group_sizes = groups.size()
group_pos = groups["label"].sum()
group_pos_rate = group_pos / group_sizes

n_groups = len(group_sizes)
print(f"\nNumber of groups (JDs): {n_groups}")
print(f"Rows per group  - min: {group_sizes.min()}, mean: {group_sizes.mean():.1f}, max: {group_sizes.max()}")
print(f"Total rows: {len(df)}")
print(f"Overall positive rate: {df['label'].mean():.4f} ({df['label'].sum()} positives)")

# Positive rate per group summary
print(f"\nPositive rate per group - min: {group_pos_rate.min():.4f}, mean: {group_pos_rate.mean():.4f}, max: {group_pos_rate.max():.4f}")

# 3. Groups with no positives
no_pos_groups = group_pos[group_pos == 0].index.tolist()
print(f"\nGroups with NO positives (dropped from ranking): {len(no_pos_groups)}")
if no_pos_groups:
    print(f"  jd_id values: {no_pos_groups[:20]}{'...' if len(no_pos_groups) > 20 else ''}")

df_filtered = df[~df["jd_id"].isin(no_pos_groups)].copy()
print(f"\nAfter dropping zero-positive groups:")
print(f"  Groups remaining: {df_filtered['jd_id'].nunique()}")
print(f"  Rows remaining:   {len(df_filtered)}")
print(f"  Positive rate:    {df_filtered['label'].mean():.4f}")

print("\nDone.")
