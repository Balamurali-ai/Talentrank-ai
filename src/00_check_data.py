"""
00_check_data.py
----------------
TalentRank AI - Data Check & Inspection Script

This script automatically locates, loads, and inspects the Resume and Job Description datasets.
It handles nested subfolders after extraction and displays shape, column names, and sample text previews.
"""

import os
import glob
from pathlib import Path
import pandas as pd


def find_csv_files(data_dir: str):
    """
    Recursively searches for CSV files inside data_dir and identifies
    the Resume dataset vs. Job Description dataset automatically.
    """
    all_csvs = [Path(p) for p in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)]
    
    resume_csv = None
    jd_csv = None
    
    for csv_path in all_csvs:
        filename_lower = csv_path.name.lower()
        if "resume" in filename_lower:
            resume_csv = csv_path
        elif any(k in filename_lower for k in ["job", "jd", "description"]):
            jd_csv = csv_path

    # Fallback if naming is non-standard
    if not resume_csv or not jd_csv:
        for csv_path in all_csvs:
            if csv_path != resume_csv and csv_path != jd_csv:
                if resume_csv is None:
                    resume_csv = csv_path
                elif jd_csv is None:
                    jd_csv = csv_path

    return resume_csv, jd_csv


def inspect_dataset(df: pd.DataFrame, dataset_name: str, text_column_candidates: list):
    """
    Prints dataset shape, column list, and a 300-character snippet of sample text.
    """
    print("=" * 70)
    print(f" DATASET SUMMARY: {dataset_name}")
    print("=" * 70)
    print(f"Shape: {df.shape} (Rows: {df.shape[0]:,}, Columns: {df.shape[1]})")
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Locate best candidate column for text preview
    sample_col = None
    for col in df.columns:
        if any(cand in col.lower() for cand in text_column_candidates):
            sample_col = col
            break
            
    if not sample_col and len(df.columns) > 0:
        sample_col = df.columns[-1]

    if sample_col in df.columns:
        sample_text = str(df[sample_col].dropna().iloc[0])
        print(f"\nSample Preview (First 300 characters from column '{sample_col}'):")
        print("-" * 70)
        print(sample_text[:300])
        print("-" * 70)
    else:
        print("\nCould not locate text column for preview.")
    print("\n")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)
    
    print(f"Searching for CSV datasets in: {data_dir}\n")
    resume_csv, jd_csv = find_csv_files(data_dir)
    
    if not resume_csv or not resume_csv.exists():
        raise FileNotFoundError(f"Resume CSV file not found in {data_dir}. Please check download step.")
    if not jd_csv or not jd_csv.exists():
        raise FileNotFoundError(f"Job Description CSV file not found in {data_dir}. Please check download step.")
        
    print(f"Found Resume CSV at: {resume_csv}")
    print(f"Found Job Description CSV at: {jd_csv}\n")
    
    # Load Resumes
    print("Loading Resume dataset...")
    df_resumes = pd.read_csv(resume_csv)
    inspect_dataset(df_resumes, "Resumes Dataset", ["resume_str", "resume", "text", "description"])
    
    # Load Job Descriptions
    print("Loading Job Description dataset...")
    df_jds = pd.read_csv(jd_csv)
    inspect_dataset(df_jds, "Job Descriptions Dataset", ["job description", "job_description", "description", "text"])


if __name__ == "__main__":
    main()
