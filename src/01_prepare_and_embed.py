"""
01_prepare_and_embed.py
-----------------------
TalentRank AI - Resume Cleaning, PII Redaction, Embedding & FAISS Indexing Script

This script:
1. Loads the raw resume dataset (Resume.csv).
2. Performs PII masking using spaCy NER (PERSON entities -> [NAME]) and regex (emails -> [EMAIL], phones -> [PHONE]).
3. Cleans whitespace while preserving natural sentence structure.
4. Generates 384-d L2-normalized embeddings using sentence-transformers (all-MiniLM-L6-v2).
5. Saves clean metadata, numpy vector arrays, and a FAISS Index (IndexFlatIP) for fast vector similarity search.
6. Validates outputs and displays side-by-side comparison of PII redaction.
"""

import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import spacy
import faiss
from sentence_transformers import SentenceTransformer

# Reconfigure stdout for UTF-8 on Windows PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Regular expressions for PII redaction
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_REGEX = re.compile(
    r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
)


def load_spacy_model():
    """Loads spaCy model with non-essential pipeline components disabled for speed."""
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
    return nlp


def redact_pii_spacy_and_regex(texts: list, nlp) -> list:
    """
    Redacts PERSON entities using spaCy NER and email/phone patterns using Regex.
    Processes texts in efficient batches using nlp.pipe on CPU.
    """
    print(f"Masking PII across {len(texts):,} resumes using spaCy NER & Regex...")
    start_time = time.time()
    
    redacted_texts = []
    
    # Process with spaCy pipe for optimal CPU performance
    for doc in nlp.pipe(texts, batch_size=64):
        raw_text = doc.text
        
        # Redact PERSON entities from back-to-front to preserve character index offsets
        person_ents = sorted(
            [ent for ent in doc.ents if ent.label_ == "PERSON"],
            key=lambda e: e.start_char,
            reverse=True
        )
        for ent in person_ents:
            raw_text = raw_text[:ent.start_char] + "[NAME]" + raw_text[ent.end_char:]
            
        # Redact email addresses and phone numbers via regex
        raw_text = EMAIL_REGEX.sub("[EMAIL]", raw_text)
        raw_text = PHONE_REGEX.sub("[PHONE]", raw_text)
        
        # Basic text cleaning: normalize whitespace/newlines
        clean_text = re.sub(r'\s+', ' ', raw_text).strip()
        
        redacted_texts.append(clean_text)
        
    elapsed = time.time() - start_time
    print(f"PII redaction completed in {elapsed:.2f} seconds ({len(texts)/elapsed:.1f} resumes/sec).")
    return redacted_texts


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "Resume" / "Resume.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        # Search recursively if nested
        all_csvs = list((project_root / "data").rglob("Resume.csv"))
        if all_csvs:
            data_path = all_csvs[0]
        else:
            raise FileNotFoundError(f"Resume dataset not found at {data_path}")
            
    print(f"1. Loading Resume dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} resumes.")
    
    # 2 & 3. PII Masking & Text Cleaning
    nlp = load_spacy_model()
    raw_texts = df["Resume_str"].astype(str).tolist()
    df["Resume_clean"] = redact_pii_spacy_and_regex(raw_texts, nlp)
    
    # 4. Sentence Transformer Embeddings
    print("\n4. Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    print("   Generating resume embeddings (384-dimensional, normalized)...")
    embed_start = time.time()
    embeddings = model.encode(
        df["Resume_clean"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embed_elapsed = time.time() - embed_start
    print(f"   Embedding generation complete in {embed_elapsed:.2f} seconds.")
    print(f"   Embeddings array shape: {embeddings.shape}")
    
    # 5. Save Clean Outputs to ./models/
    print("\n5. Saving processed artifacts to ./models/...")
    
    clean_csv_path = models_dir / "resumes_clean.csv"
    embeddings_npy_path = models_dir / "resume_embeddings.npy"
    ids_npy_path = models_dir / "resume_ids.npy"
    faiss_index_path = models_dir / "resume_index.faiss"
    
    # Save CSV without raw Resume_str (PII safe)
    df[["ID", "Category", "Resume_clean"]].to_csv(clean_csv_path, index=False)
    np.save(embeddings_npy_path, embeddings.astype('float32'))
    np.save(ids_npy_path, df["ID"].values)
    
    # 6. Build and Save FAISS Index (Cosine Similarity / IndexFlatIP)
    print("\n6. Building FAISS vector index (IndexFlatIP)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, str(faiss_index_path))
    
    # 7. Validation & Summary
    print("\n" + "=" * 70)
    print(" STEP 2 SUMMARY & VALIDATION")
    print("=" * 70)
    print(f"Total Resumes Processed: {len(df):,}")
    print(f"Embedding Shape:        {embeddings.shape}")
    print(f"Embedding Time:         {embed_elapsed:.2f}s ({len(df)/embed_elapsed:.1f} resumes/sec)")
    print("\nSaved Artifacts & File Sizes:")
    for path in [clean_csv_path, embeddings_npy_path, ids_npy_path, faiss_index_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path.name:22s}: {size_mb:.2f} MB")
        
    print("\n" + "=" * 70)
    print(" PII MASKING VISUAL COMPARISON (First 2 Samples)")
    print("=" * 70)
    for idx in range(min(2, len(df))):
        orig_snippet = raw_texts[idx][:300].replace('\n', ' ')
        clean_snippet = df["Resume_clean"].iloc[idx][:300]
        print(f"\n--- Sample {idx + 1} (Category: {df['Category'].iloc[idx]}) ---")
        print(f"RAW (First 300 chars):\n  {orig_snippet}")
        print(f"CLEAN & MASKED (First 300 chars):\n  {clean_snippet}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
