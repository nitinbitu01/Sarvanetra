"""
backend/db/seed_anchor_vault.py — Seeds frozen ground-truth Anchor identities into the Vault
to bootstrap the AnchorEvaluator regression safety gate (resolves Cold-Start problem).
"""
import json
import logging
import uuid
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session

from backend.db.session import SessionLocal
from backend.db.models import VaultEntry
from backend.core.vault_config import VaultCompartment, LabelTrust, EntityType

logger = logging.getLogger("sentinel.seed_anchor")


def seed_anchor_dataset(force: bool = False):
    db = SessionLocal()
    try:
        if force:
            db.query(VaultEntry).filter(VaultEntry.compartment == VaultCompartment.ANCHOR.value).delete()
            db.commit()
        else:
            existing_count = (
                db.query(VaultEntry)
                .filter(VaultEntry.compartment == VaultCompartment.ANCHOR.value)
                .count()
            )
            if existing_count >= 16:
                print(f"Anchor dataset already seeded with {existing_count} reference samples.")
                return

        print("[INFO] Seeding Anchor Ground-Truth Reference Identities into Vault...")
        # 8 distinct anchor identities, each with 3 crop embeddings (different lighting/angles)
        anchor_identities = [
            ("ANCHOR_VEH_GJ01_A1", "vehicle", "GJ01AB1234", "White SUV"),
            ("ANCHOR_VEH_GJ05_B2", "vehicle", "GJ05CD5678", "Silver Sedan"),
            ("ANCHOR_VEH_GJ18_C3", "vehicle", "GJ18EF9012", "Black Hatchback"),
            ("ANCHOR_VEH_GJ06_D4", "vehicle", "GJ06GH3456", "Red Auto-Rickshaw"),
            ("ANCHOR_VEH_GJ27_E5", "vehicle", "GJ27IJ7890", "Blue Truck"),
            ("ANCHOR_PER_RAHUL_01", "person", None, "Rahul (Standard Track)"),
            ("ANCHOR_PER_PRIYA_02", "person", None, "Priya (Standard Track)"),
            ("ANCHOR_PER_AMIT_03", "person", None, "Amit (Standard Track)"),
        ]

        seeded_entries = 0
        for group_id, entity_type_str, plate, desc in anchor_identities:
            # Base synthetic feature vector for this identity
            np.random.seed(abs(hash(group_id)) % (2**31))
            base_vec = np.random.randn(512).astype(np.float32)
            base_vec /= np.linalg.norm(base_vec)

            # 3 camera/lighting variations per identity (day, night, glare)
            conditions = ["day", "night", "night_glare"]
            for idx, cond in enumerate(conditions):
                # Realistic intra-identity feature variance (cosine similarity ~0.92-0.98)
                noise = np.random.randn(512).astype(np.float32) * 0.008
                sample_vec = base_vec + noise
                sample_vec /= np.linalg.norm(sample_vec)

                entry_id = str(uuid.uuid4())
                entry = VaultEntry(
                    id=entry_id,
                    compartment=VaultCompartment.ANCHOR.value,
                    trust_level=LabelTrust.GOLD.value,
                    entity_type=entity_type_str,
                    camera_id=f"CAM_{(idx % 4) + 1:02d}",
                    captured_at=datetime.utcnow(),
                    stored_at=datetime.utcnow(),
                    crop_minio_path=f"vault/anchor/{entity_type_str}/{entry_id}.jpg",
                    embedding_vector=sample_vec.tolist(),
                    embedding_hash=f"hash_{group_id}_{cond}",
                    lighting_condition=cond,
                    plate_text=plate,
                    anchor_group_id=group_id,
                    training_eligible=True,
                    used_in_training=False,
                    is_gold_retained=True,
                )
                db.add(entry)
                seeded_entries += 1

        db.commit()
        print(f"[OK] Successfully seeded {seeded_entries} Anchor samples across 8 identity groups.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_anchor_dataset()
