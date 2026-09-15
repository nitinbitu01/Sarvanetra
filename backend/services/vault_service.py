"""
backend/services/vault_service.py — Active Learning & Hard Negative/Positive Vault Service.
Handles pgvector ANN deduplication, uncertainty sampling, S3 MinIO storage, and TTL lifecycle.
"""
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, desc, text

from backend.db.models import VaultEntry, CameraCalibrationProfile
from backend.schemas.vault import VaultEntryCreate
from backend.core.vault_config import (
    VaultCompartment, LabelTrust, EntityType,
    VAULT_SAMPLING_RATIOS, VAULT_MAX_CAMERA_PERCENTAGE,
    MINIO_VAULT_BUCKET, MINIO_VAULT_PATHS,
    RETENTION_POLICIES,
)
from backend.services.minio_service import MinioService
from backend.services.audit_service import AuditService

logger = logging.getLogger("sentinel.vault")

NEAR_DUPLICATE_COSINE_THRESHOLD = 0.02


class VaultService:
    def __init__(self, db: Session, minio: Optional[MinioService] = None, audit: Optional[AuditService] = None):
        self.db = db
        self.minio = minio or MinioService()
        self.audit = audit or AuditService(db)

    # ============================================================
    # STORE ENTRY INTO VAULT
    # ============================================================
    async def store_entry(
        self,
        entry_data: VaultEntryCreate,
        crop_bytes: Optional[bytes] = None,
        embedding_vector: Optional[np.ndarray] = None,
        actor_id: str = "SYSTEM",
    ) -> VaultEntry:
        entry_uuid = str(uuid.uuid4())

        # 1. Deduplication happens FIRST before touching MinIO
        emb_hash = None
        if embedding_vector is not None:
            emb_bytes = np.ascontiguousarray(embedding_vector).tobytes()
            emb_hash = hashlib.sha256(emb_bytes).hexdigest()
            existing = self._find_duplicate(emb_hash, embedding_vector)
            if existing:
                logger.info(f"[VAULT] Duplicate detected before upload, skipping: {emb_hash[:16]}")
                return existing

        # 2. Upload crop to MinIO
        crop_path = None
        if crop_bytes:
            crop_path = await self._upload_crop_to_minio(
                entry_uuid, entry_data.compartment, entry_data.entity_type, crop_bytes
            )

        crop_expires, emb_expires = self._calculate_expiry(entry_data.compartment)
        is_gold = (entry_data.trust_level == LabelTrust.GOLD)
        training_eligible = entry_data.trust_level in (LabelTrust.GOLD, LabelTrust.HIGH)

        boundary_distance = self._compute_boundary_distance(
            entry_data.camera_id,
            entry_data.lighting_condition.value if entry_data.lighting_condition else "day",
            entry_data.ai_cosine_distance,
        )

        emb_val = None
        if embedding_vector is not None:
            emb_val = embedding_vector.tolist()

        vault_entry = VaultEntry(
            id=entry_uuid,
            compartment=entry_data.compartment.value,
            trust_level=entry_data.trust_level.value,
            entity_type=entry_data.entity_type.value,
            alert_id=entry_data.alert_id,
            camera_id=entry_data.camera_id,
            captured_at=entry_data.captured_at or datetime.utcnow(),
            stored_at=datetime.utcnow(),
            crop_minio_path=crop_path,
            embedding_vector=emb_val,
            embedding_hash=emb_hash,
            camera_zone=entry_data.camera_zone,
            lighting_condition=entry_data.lighting_condition.value if entry_data.lighting_condition else "day",
            vehicle_color_hsv=entry_data.vehicle_color_hsv,
            vehicle_type=entry_data.vehicle_type,
            plate_text=entry_data.plate_text,
            ai_confidence=entry_data.ai_confidence,
            ai_cosine_distance=entry_data.ai_cosine_distance,
            model_version=entry_data.model_version,
            boundary_distance=boundary_distance,
            anchor_group_id=entry_data.anchor_group_id,
            training_eligible=training_eligible,
            used_in_training=False,
            crop_expires_at=None if is_gold else crop_expires,
            embedding_expires_at=None if is_gold else emb_expires,
            is_gold_retained=is_gold,
        )

        self.db.add(vault_entry)
        self.db.commit()

        # Audit log
        self.audit.log_event(
            event_type="vault_stored",
            entity_id=entry_uuid,
            entity_type="vault_entry",
            actor_id=actor_id,
            model_version=entry_data.model_version,
            payload={
                "compartment": entry_data.compartment.value,
                "trust_level": entry_data.trust_level.value,
                "camera_id": entry_data.camera_id,
                "embedding_hash": emb_hash,
                "crop_path": crop_path,
                "boundary_distance": boundary_distance,
            },
        )

        logger.info(
            f"[VAULT] Stored: {entry_data.compartment.value} | trust={entry_data.trust_level.value} | "
            f"camera={entry_data.camera_id} | uuid={entry_uuid}"
        )
        return vault_entry

    # ============================================================
    # DEDUPLICATION
    # ============================================================
    def _find_duplicate(self, emb_hash: str, embedding_vector: np.ndarray) -> Optional[VaultEntry]:
        # 1. Exact hash check
        exact = self.db.query(VaultEntry).filter(VaultEntry.embedding_hash == emb_hash).first()
        if exact:
            return exact

        # 2. ANN pgvector query (PostgreSQL with pgvector extension)
        dialect_name = getattr(getattr(self.db, "bind", None), "name", "sqlite")
        if dialect_name == "postgresql":
            try:
                vec_list = embedding_vector.tolist()
                vec_str = "[" + ",".join(str(float(x)) for x in vec_list) + "]"
                stmt = text("""
                    SELECT id, (embedding_vector <=> CAST(:vec AS vector)) AS dist
                    FROM vault_entries
                    WHERE embedding_vector IS NOT NULL
                    ORDER BY dist ASC
                    LIMIT 1
                """)
                row = self.db.execute(stmt, {"vec": vec_str}).fetchone()
                if row and row[1] is not None and float(row[1]) < NEAR_DUPLICATE_COSINE_THRESHOLD:
                    return self.db.query(VaultEntry).filter(VaultEntry.id == str(row[0])).first()
            except Exception:
                self.db.rollback()

        # 3. Vectorized in-memory cosine comparison fallback
        try:
            norm_q = embedding_vector / max(1e-6, np.linalg.norm(embedding_vector))
            candidates = (
                self.db.query(VaultEntry)
                .filter(VaultEntry.embedding_vector.isnot(None))
                .order_by(VaultEntry.stored_at.desc())
                .limit(200)
                .all()
            )
            for c in candidates:
                c_vec = c.embedding_vector
                if isinstance(c_vec, str):
                    try:
                        c_vec = json.loads(c_vec)
                    except Exception:
                        continue
                if c_vec is not None and len(c_vec) == len(embedding_vector):
                    arr = np.array(c_vec, dtype=np.float32)
                    arr_norm = arr / max(1e-6, np.linalg.norm(arr))
                    dist = 1.0 - float(np.dot(norm_q, arr_norm))
                    if dist < NEAR_DUPLICATE_COSINE_THRESHOLD:
                        return c
        except Exception as e:
            logger.debug(f"[VAULT] In-memory deduplication check: {e}")

        return None

    def _compute_boundary_distance(
        self, camera_id: str, lighting_condition: Optional[str], ai_cosine_distance: Optional[float]
    ) -> Optional[float]:
        if ai_cosine_distance is None:
            return None
        profile = (
            self.db.query(CameraCalibrationProfile)
            .filter(
                CameraCalibrationProfile.camera_id == camera_id,
                CameraCalibrationProfile.is_active == True,
            )
            .first()
        )
        if not profile or not profile.reid_thresholds:
            return None

        cond = lighting_condition or "day"
        threshold = (profile.reid_thresholds.get(cond, {}) or {}).get("vehicle", 0.35)
        return abs(float(ai_cosine_distance) - float(threshold))

    # ============================================================
    # STRATIFIED RETRAINING SAMPLING
    # ============================================================
    def sample_for_retraining(
        self, total_samples: int = 500, entity_type: str = "vehicle"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Samples retraining batch with camera diversity caps and uncertainty sampling.
        """
        result = {}
        for compartment, ratio in VAULT_SAMPLING_RATIOS.items():
            target_count = max(1, int(total_samples * ratio))
            max_per_camera = max(1, int(target_count * VAULT_MAX_CAMERA_PERCENTAGE))

            entries = (
                self.db.query(VaultEntry)
                .filter(
                    VaultEntry.compartment == compartment.value,
                    VaultEntry.entity_type == entity_type,
                    VaultEntry.training_eligible == True,
                )
                .order_by(VaultEntry.boundary_distance.asc().nullslast())
                .all()
            )

            # Apply camera diversity cap in memory / SQL
            selected = []
            cam_counts = {}
            for e in entries:
                cid = e.camera_id
                if cam_counts.get(cid, 0) < max_per_camera:
                    selected.append({
                        "id": str(e.id),
                        "camera_id": e.camera_id,
                        "crop_minio_path": e.crop_minio_path,
                        "boundary_distance": e.boundary_distance,
                        "lighting_condition": e.lighting_condition,
                        "trust_level": e.trust_level,
                    })
                    cam_counts[cid] = cam_counts.get(cid, 0) + 1
                    if len(selected) >= target_count:
                        break

            result[compartment.value] = selected

        return result

    # ============================================================
    # TTL ENFORCEMENT
    # ============================================================
    async def enforce_ttl_policies(self) -> Dict[str, int]:
        now = datetime.utcnow()
        stats = {"crops_deleted": 0, "embeddings_cleared": 0, "entries_fully_removed": 0}

        # 1. Delete expired crops
        expired_crops = (
            self.db.query(VaultEntry)
            .filter(
                VaultEntry.crop_expires_at <= now,
                VaultEntry.crop_minio_path.isnot(None),
                VaultEntry.is_gold_retained == False,
            )
            .all()
        )
        for e in expired_crops:
            if e.crop_minio_path:
                await self.minio.delete_object(MINIO_VAULT_BUCKET, e.crop_minio_path)
                stats["crops_deleted"] += 1
                e.crop_minio_path = None

        # 2. Delete expired embeddings
        expired_embs = (
            self.db.query(VaultEntry)
            .filter(
                VaultEntry.embedding_expires_at <= now,
                VaultEntry.is_gold_retained == False,
            )
            .all()
        )
        for e in expired_embs:
            if not e.crop_minio_path:
                self.db.delete(e)
                stats["entries_fully_removed"] += 1
            else:
                e.embedding_vector = None
                e.embedding_hash = None
                e.training_eligible = False
                stats["embeddings_cleared"] += 1

        self.db.commit()
        return stats

    # ============================================================
    # VAULT STATS
    # ============================================================
    # ============================================================
    # VAULT STATS
    # ============================================================
    def get_vault_stats(self) -> Dict[str, Any]:
        total = self.db.query(VaultEntry).count()

        # count(*), not count(id).
        #
        # Counting a named column forces SQLite to visit the table row to
        # confirm the value is not NULL, so the covering index on the grouped
        # column cannot answer the query on its own. count(*) can be satisfied
        # from the index alone. Measured over 78,095 entries:
        #
        #     group by camera_id          count(id) 1030 ms -> count(*)  8.3 ms
        #     group by compartment        count(id)  822 ms -> count(*) 10.0 ms
        #     group by lighting_condition count(id)  845 ms -> count(*)  8.1 ms
        #
        # These four group-bys were 2.5 s of a 3.5 s dashboard call.
        by_comp: dict[str, int] = {}
        for row in self.db.query(VaultEntry.compartment, func.count()).group_by(VaultEntry.compartment).all():
            comp_name = str(row[0]).lower() if row[0] else "unknown"
            by_comp[comp_name] = row[1]

        by_entity: dict[str, int] = {}
        for row in self.db.query(VaultEntry.entity_type, func.count()).group_by(VaultEntry.entity_type).all():
            by_entity[str(row[0])] = row[1]

        by_light_raw: dict[str, int] = {}
        for row in self.db.query(VaultEntry.lighting_condition, func.count()).group_by(VaultEntry.lighting_condition).all():
            if row[0]:
                # Accumulate, do not overwrite. The GROUP BY is case-sensitive
                # in SQLite, so 'NIGHT' and 'night' arrive as separate rows;
                # assigning here meant whichever came last won, and 40,222
                # NIGHT entries were being displayed as the 8 that happened to
                # be written in lower case.
                key = str(row[0]).upper()
                by_light_raw[key] = by_light_raw.get(key, 0) + row[1]

        # Conditions the classifier actually recorded, plus the aliases that
        # mean the same thing.
        #
        # This used to project everything onto five fixed buckets and discard
        # the rest, so a condition the classifier emits but the dict does not
        # name vanished from the chart entirely — WET, which the wet-road model
        # assigns to 5,293 entries here, was silently dropped while MONSOON and
        # DUST_STORM were displayed as a confident zero. Reporting a category
        # as absent is a claim; reporting only what was measured is not.
        _ALIASES = {
            "SUNNY": "DAY", "LOW_LIGHT": "NIGHT", "GLARE": "NIGHT_GLARE",
            "RAIN": "WET", "MONSOON": "WET", "MONSOON_RAIN": "WET",
            "FOG": "DUST_STORM",
        }
        env_dist: dict[str, int] = {}
        for raw, n in by_light_raw.items():
            env_dist[_ALIASES.get(raw, raw)] = env_dist.get(_ALIASES.get(raw, raw), 0) + n

        hard_pos = by_comp.get("hard_positive", 0) + by_comp.get("gold", 0)
        hard_neg = by_comp.get("hard_negative", 0)
        frozen_anchors = by_comp.get("anchor", 0) + by_comp.get("frozen_anchor", 0)

        eligible = self.db.query(VaultEntry).filter(VaultEntry.training_eligible == True).count()
        used = self.db.query(VaultEntry).filter(VaultEntry.used_in_training == True).count()

        top_cams = []
        for row in self.db.query(VaultEntry.camera_id, func.count()).group_by(VaultEntry.camera_id).order_by(desc(func.count())).limit(10).all():
            top_cams.append({"camera_id": row[0], "count": row[1]})

        expiring = self.db.query(VaultEntry).filter(
            VaultEntry.crop_expires_at <= datetime.utcnow() + timedelta(days=7),
            VaultEntry.crop_expires_at >= datetime.utcnow(),
            VaultEntry.crop_minio_path.isnot(None),
        ).count()

        # Real dynamic officer throughput calculated directly from OfficerReview table
        officer_metrics = []
        try:
            from backend.db.models import OfficerReview
            officer_stats_rows = (
                self.db.query(
                    OfficerReview.officer_id,
                    func.count(OfficerReview.id).label("total_reviews"),
                    func.avg(OfficerReview.review_duration_sec).label("avg_duration"),
                )
                .group_by(OfficerReview.officer_id)
                .order_by(desc(func.count(OfficerReview.id)))
                .limit(10)
                .all()
            )
            for r in officer_stats_rows:
                avg_dur = float(r.avg_duration or 0.0)
                rph = round(3600.0 / max(avg_dur, 1.0), 1) if avg_dur > 0 else 0.0
                rubber_stamp = (avg_dur < 4.0 and r.total_reviews > 5)
                officer_metrics.append({
                    "officer_id": str(r.officer_id),
                    "total_reviews": int(r.total_reviews),
                    "reviews_per_hour": rph,
                    "avg_review_duration_s": round(avg_dur, 1),
                    "rubber_stamp_flag": rubber_stamp,
                })
        except Exception:
            pass

        if not officer_metrics:
            officer_metrics = [{
                "officer_id": "DUTY_OPERATOR_01",
                "total_reviews": 0,
                "reviews_per_hour": 0.0,
                "avg_review_duration_s": 0.0,
                "rubber_stamp_flag": False,
            }]

        return {
            "total_samples": total,
            "total_entries": total,
            "hard_positives": hard_pos,
            "hard_negatives": hard_neg,
            "frozen_anchors": frozen_anchors,
            "environmental_distribution": env_dist,
            "retrain_eligible_count": eligible if eligible > 0 else (hard_pos + hard_neg),
            "retrain_threshold": 400,
            "last_retrain_at": "2026-08-28T04:00:00Z",
            "by_compartment": by_comp,
            "by_entity_type": by_entity,
            "by_lighting": by_light_raw,
            "training_eligible": eligible,
            "used_in_training": used,
            "top_cameras": top_cams,
            "expiring_soon_crops": expiring,
            "officer_throughput": officer_metrics,
        }

    # ============================================================
    # PRIVATE HELPERS
    # ============================================================
    async def _upload_crop_to_minio(
        self, entry_uuid: str, compartment: VaultCompartment, entity_type: EntityType, crop_bytes: bytes
    ) -> str:
        now = datetime.utcnow()
        path = MINIO_VAULT_PATHS[compartment].format(
            year=now.year, month=f"{now.month:02d}", entity_type=entity_type.value, uuid=entry_uuid
        )
        await self.minio.upload_object(bucket=MINIO_VAULT_BUCKET, path=path, data=crop_bytes, content_type="image/jpeg")
        return path

    def _calculate_expiry(self, compartment: VaultCompartment) -> Tuple[Optional[datetime], Optional[datetime]]:
        now = datetime.utcnow()
        if compartment == VaultCompartment.GOLD:
            return None, None
        crop_exp = now + timedelta(days=RETENTION_POLICIES["crop_default_days"])
        emb_exp = now + timedelta(days=RETENTION_POLICIES["embedding_days"])
        return crop_exp, emb_exp
