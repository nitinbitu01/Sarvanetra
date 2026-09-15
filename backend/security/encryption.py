import os
import hashlib
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _get_key() -> bytes:
    raw = os.getenv("EVIDENCE_AES_KEY", "32-byte-sentinel-gujarat-key-!")
    if len(raw) < 32:
        raw = raw.ljust(32, "0")
    return raw[:32].encode("utf-8")


def encrypt_file(source_path: str, dest_path: str) -> str:
    """
    Encrypt file using AES-256-GCM.
    Returns SHA-256 hash of the ORIGINAL (plaintext) file.
    """
    key     = _get_key()
    aesgcm  = AESGCM(key)
    nonce   = os.urandom(12)

    with open(source_path, "rb") as f:
        plaintext = f.read()

    original_hash = hashlib.sha256(plaintext).hexdigest()
    ciphertext    = aesgcm.encrypt(nonce, plaintext, None)

    with open(dest_path, "wb") as f:
        f.write(nonce + ciphertext)

    return original_hash


def decrypt_file(source_path: str, dest_path: str) -> None:
    """Decrypt AES-256-GCM encrypted file"""
    key    = _get_key()
    aesgcm = AESGCM(key)

    with open(source_path, "rb") as f:
        data = f.read()

    nonce      = data[:12]
    ciphertext = data[12:]
    plaintext  = aesgcm.decrypt(nonce, ciphertext, None)

    with open(dest_path, "wb") as f:
        f.write(plaintext)


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
