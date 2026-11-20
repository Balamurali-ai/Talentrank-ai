"""
07_encrypt_resumes.py
---------------------
TalentRank AI - Step 7: Security Hardening

Encrypts the plaintext resumes_clean.csv at rest into an encrypted .enc file,
and removes the original plaintext CSV.
"""

import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv, set_key

def main():
    print("=" * 50)
    print(" STEP 7: ENCRYPTING DATA AT REST")
    print("=" * 50)
    
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    
    load_dotenv(env_path)
    
    key_str = os.getenv("ENCRYPTION_KEY")
    if not key_str:
        print("ENCRYPTION_KEY not found in .env. Generating a new one...")
        key_str = Fernet.generate_key().decode()
        if not env_path.exists():
            env_path.touch()
        set_key(str(env_path), "ENCRYPTION_KEY", key_str)
        print(f"Added ENCRYPTION_KEY to {env_path}")
        
    fernet = Fernet(key_str.encode())
    
    models_dir = project_root / "models"
    csv_path = models_dir / "resumes_clean.csv"
    enc_path = models_dir / "resumes_clean.enc"
    
    if not csv_path.exists():
        if enc_path.exists():
            print(f"{csv_path} does not exist, but {enc_path} is already present.")
            return
        else:
            print(f"Error: {csv_path} not found.")
            return
            
    print(f"Encrypting {csv_path}...")
    with open(csv_path, "rb") as f:
        plaintext = f.read()
        
    encrypted = fernet.encrypt(plaintext)
    
    with open(enc_path, "wb") as f:
        f.write(encrypted)
        
    print(f"Successfully encrypted to {enc_path}")
    
    # Removing the plaintext file
    print("Removing plaintext file (this is destructive!)...")
    os.remove(csv_path)
    print("Plaintext file removed successfully.")

if __name__ == "__main__":
    main()
