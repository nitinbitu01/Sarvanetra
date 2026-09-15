"""
backend/scripts/seed_vault_data.py — Seed Active Learning Vault with Realistic Gujarat CCTV Intelligence

Generates a production-grade active learning dataset across 30 Gujarat cameras,
covering all 5 Gujarat environmental conditions (Day, Night, Night Glare, Monsoon, Dust Storm),
with Hard Positives, Hard Negatives, Frozen Anchors, and Officer Review audit logs.
"""
import uuid
import random
from datetime import datetime, timedelta
from backend.db.session import SessionLocal
from backend.db.models import VaultEntry, OfficerReview, Alert, Camera

ENVIRONMENTS = ["DAY", "NIGHT", "NIGHT_GLARE", "MONSOON", "DUST_STORM"]
ENV_WEIGHTS = [0.38, 0.28, 0.16, 0.12, 0.06]

COMPARTMENTS = ["hard_positive", "hard_negative", "anchor", "gold", "probationary"]
COMP_WEIGHTS = [0.42, 0.30, 0.16, 0.08, 0.04]

VEHICLE_TYPES = ["Sedan", "SUV", "Motorcycle", "Autorickshaw", "Commercial Truck", "Bus"]
COLORS = ["White", "Silver", "Black", "Red", "Blue", "Yellow", "Grey"]

CAMERAS = [f"CAM_{i:02d}" for i in range(1, 31)]

OFFICERS = [
    ("OFFICER_PATEL_402", "Inspector Rajesh Patel", 8.4, 34.2, False),
    ("OFFICER_DESAI_108", "Sub-Inspector Priya Desai", 9.1, 28.5, False),
    ("OFFICER_JADEJA_220", "Constable Vikram Jadeja", 7.8, 38.0, False),
    ("OFFICER_MEHTA_305", "Operator Amit Mehta", 11.2, 22.0, False),
]

def seed_vault(target_count: int = 498):
    db = SessionLocal()
    try:
        existing_count = db.query(VaultEntry).count()
        if existing_count >= target_count:
            print(f"Vault already seeded with {existing_count} entries. Skipping.")
            return

        print(f"Seeding {target_count} Active Learning Vault entries...")
        now = datetime.utcnow()
        entries = []
        reviews = []

        # Get existing alert IDs if any
        alert_ids = [r[0] for r in db.query(Alert.id).limit(100).all()]

        for i in range(target_count):
            entry_id = str(uuid.uuid4())
            comp = random.choices(COMPARTMENTS, weights=COMP_WEIGHTS)[0]
            env = random.choices(ENVIRONMENTS, weights=ENV_WEIGHTS)[0]
            cam_id = random.choice(CAMERAS)
            entity_type = random.choice(["vehicle", "person"])
            
            captured_at = now - timedelta(days=random.randint(0, 14), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            
            # Confidence & distance
            if comp in ("hard_positive", "gold"):
                ai_conf = round(random.uniform(0.45, 0.72), 3) # Hard cases have moderate initial confidence
                trust_level = "gold" if comp == "gold" else "high"
                training_eligible = True
            elif comp == "hard_negative":
                ai_conf = round(random.uniform(0.50, 0.85), 3) # False alarms initially had false high confidence
                trust_level = "verified_negative"
                training_eligible = True
            elif comp == "anchor":
                ai_conf = round(random.uniform(0.88, 0.98), 3) # Benchmark anchors have high confidence
                trust_level = "gold"
                training_eligible = True
            else:
                ai_conf = round(random.uniform(0.30, 0.60), 3)
                trust_level = "probationary"
                training_eligible = False

            ai_dist = round(random.uniform(0.12, 0.48), 3)
            boundary_dist = round(abs(ai_dist - 0.35), 4)

            entry = VaultEntry(
                id=entry_id,
                compartment=comp,
                trust_level=trust_level,
                entity_type=entity_type,
                alert_id=random.choice(alert_ids) if alert_ids and random.random() < 0.6 else None,
                camera_id=cam_id,
                captured_at=captured_at,
                stored_at=captured_at + timedelta(minutes=random.randint(1, 10)),
                lighting_condition=env,
                crop_minio_path=f"vault/{comp}/{captured_at.year}/{captured_at.month:02d}/{entity_type}/{entry_id}.jpg",
                camera_zone=f"Zone_{cam_id.split('_')[-1]}",
                vehicle_type=random.choice(VEHICLE_TYPES) if entity_type == "vehicle" else None,
                plate_text=f"GJ{random.randint(1, 38):02d}AA{random.randint(1000, 9999)}" if entity_type == "vehicle" else None,
                ai_confidence=ai_conf,
                ai_cosine_distance=ai_dist,
                boundary_distance=boundary_dist,
                model_version="yolov8s_osnet_v2.4",
                training_eligible=training_eligible,
                used_in_training=(random.random() < 0.28 if training_eligible else False),
                is_gold_retained=(comp in ("gold", "anchor")),
                crop_expires_at=(None if comp in ("gold", "anchor") else captured_at + timedelta(days=90)),
                embedding_expires_at=(None if comp in ("gold", "anchor") else captured_at + timedelta(days=180)),
            )
            entries.append(entry)

            # Create corresponding officer review for reviewed entries
            if comp in ("hard_positive", "hard_negative", "gold"):
                officer_meta = random.choice(OFFICERS)
                dur = round(random.uniform(max(3.0, officer_meta[2] - 3.0), officer_meta[2] + 4.0), 2)
                rev = OfficerReview(
                    id=str(uuid.uuid4()),
                    review_session_id=str(uuid.uuid4()),
                    vault_entry_id=entry_id,
                    alert_id=entry.alert_id,
                    officer_id=officer_meta[0],
                    officer_role="OFFICER",
                    decision="APPROVE" if comp in ("hard_positive", "gold") else "REJECT",
                    review_duration_sec=dur,
                    client_reported_duration_sec=dur,
                    confidence_shown=False,
                    review_started_at=captured_at + timedelta(minutes=5),
                    review_completed_at=captured_at + timedelta(minutes=5, seconds=dur),
                    created_at=captured_at + timedelta(minutes=5),
                )
                reviews.append(rev)

        db.bulk_save_objects(entries)
        if reviews:
            db.bulk_save_objects(reviews)
        db.commit()

        print(f"Successfully seeded {len(entries)} VaultEntry rows and {len(reviews)} OfficerReview rows!")
        
        # Verify stats
        from backend.services.vault_service import VaultService
        svc = VaultService(db=db)
        stats = svc.get_vault_stats()
        print("\nGenerated Vault Statistics Summary:")
        print(f"  - Total Samples:      {stats['total_samples']}")
        print(f"  - Hard Positives:     {stats['hard_positives']}")
        print(f"  - Hard Negatives:     {stats['hard_negatives']}")
        print(f"  - Frozen Anchors:     {stats['frozen_anchors']}")
        print(f"  - Retrain Readiness:  {stats['retrain_eligible_count']} / {stats['retrain_threshold']} ({round(stats['retrain_eligible_count'] / stats['retrain_threshold'] * 100, 1)}%)")
        print(f"  - Gujarat Weather:    {stats['environmental_distribution']}")

    except Exception as e:
        db.rollback()
        print(f"Error seeding vault: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_vault()
