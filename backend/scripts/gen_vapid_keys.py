"""
backend/scripts/gen_vapid_keys.py — generate a VAPID keypair for Web Push.

Run ONCE. Put the output in .env. Never regenerate per request, and never
commit the private key.

    python -m backend.scripts.gen_vapid_keys

KEY ROTATION (documented, not automated)
  Rotating VAPID keys invalidates EVERY existing push subscription — the
  browser signed each one against the old public key and the push service
  will reject sends with the new one. There is no server-side migration; the
  devices themselves must re-subscribe. When rotation is required:
    1. Generate a new pair here.
    2. Update VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY in .env and restart.
    3. Existing sends start returning 401/403 from the push service; the
       send path already logs those to push_delivery_log rather than
       crashing, so the failure is visible rather than silent.
    4. Officers must re-open the PWA and tap Enable Alerts again.
  Rotation is therefore a user-facing event, not a config change. Plan it.
"""
from __future__ import annotations

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _b64url(raw: bytes) -> str:
    """URL-safe base64 with padding stripped — the form browsers expect."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate() -> tuple[str, str]:
    """Return (public_key, private_key) as URL-safe base64 strings.

    The public key is the uncompressed P-256 point (65 bytes, 0x04-prefixed)
    that pushManager.subscribe() expects as applicationServerKey. The private
    key is the raw 32-byte scalar, which is what pywebpush accepts as
    vapid_private_key when given as a base64url string.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())

    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    private_raw = private_key.private_numbers().private_value.to_bytes(32, "big")

    return _b64url(public_raw), _b64url(private_raw)


if __name__ == "__main__":
    pub, priv = generate()
    print("\nAdd these to .env (do NOT commit the private key):\n")
    print(f"VAPID_PUBLIC_KEY={pub}")
    print(f"VAPID_PRIVATE_KEY={priv}")
    print(f"\nPublic key length: {len(pub)} chars (expect 87 for P-256)")
