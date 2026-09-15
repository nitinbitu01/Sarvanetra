"""day6_reid_tables — GlobalPerson, ReIDReviewItem, ReIDCalibrationLog, Journey extended.

Hand-crafted for SQLite compatibility (same pattern as Day 5 migration).
SQLite cannot ALTER TABLE to add NOT NULL columns without defaults, so all
new columns on existing tables are nullable or carry server_default.

New tables:
  - global_persons          (GlobalPerson canonical identity records)
  - reid_review_items       (human review queue for borderline decisions)
  - reid_calibration_log    (every resolved decision for threshold recalibration)

Modified table:
  - journeys                (adds Day 6 ReID columns; legacy columns stay)
"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = '3ae3fabcabe8'   # Day 5 migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── New table: reid_calibration_log (created first — ReIDReviewItem FKs it) ──
    op.create_table(
        'reid_calibration_log',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('decision', sa.String(length=32), nullable=False),
        sa.Column('was_correct', sa.Boolean(), nullable=True),
        sa.Column('time_gap_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )

    # ── New table: global_persons ─────────────────────────────────────────────
    op.create_table(
        'global_persons',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('representative_embedding', sa.LargeBinary(), nullable=True),
        sa.Column('faiss_index_position', sa.Integer(), nullable=True),
        sa.Column('first_seen_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.Column('last_seen_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.Column('total_sightings', sa.Integer(), nullable=False,
                  server_default='1'),
        sa.Column('retention_expires_at', sa.DateTime(), nullable=True),
        sa.Column('retention_hold', sa.Boolean(), nullable=False,
                  server_default='0'),
        sa.Column('legal_basis', sa.String(length=256), nullable=False,
                  server_default='routine_public_safety_monitoring'),
        sa.Column('is_deleted', sa.Boolean(), nullable=False,
                  server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('faiss_index_position'),
    )

    # ── New table: reid_review_items ─────────────────────────────────────────
    op.create_table(
        'reid_review_items',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('local_track_id', sa.Integer(), nullable=True),
        sa.Column('candidate_global_person_id', sa.Integer(), nullable=True),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('crop_image_path', sa.String(length=512), nullable=True),
        sa.Column('candidate_reference_image_path', sa.String(length=512),
                  nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False,
                  server_default='PENDING'),
        sa.Column('reviewed_by_user_id', sa.Integer(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('time_gap_seconds', sa.Integer(), nullable=True),
        sa.Column('calibration_log_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['local_track_id'], ['tracks.id']),
        sa.ForeignKeyConstraint(['candidate_global_person_id'], ['global_persons.id']),
        sa.ForeignKeyConstraint(['reviewed_by_user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['calibration_log_id'], ['reid_calibration_log.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_reid_review_items_status', 'reid_review_items',
                    ['status'], unique=False)

    # ── Extend journeys table with Day 6 ReID columns ────────────────────────
    # NOTE: legacy columns (global_id, camera_id string, track_id) are kept
    # exactly as-is — only new nullable columns added.
    with op.batch_alter_table('journeys') as batch_op:
        batch_op.add_column(
            sa.Column('global_person_id', sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column('camera_db_id', sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column('local_track_id', sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column('seen_at', sa.DateTime(), nullable=True))
        batch_op.add_column(
            sa.Column('confidence', sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column('confidence_caveat', sa.String(length=512), nullable=True))
        batch_op.add_column(
            sa.Column('created_at', sa.DateTime(), nullable=True,
                      server_default=sa.func.now()))

    # Index for fast journey lookups by global_person_id
    op.create_index('ix_journeys_global_person_id', 'journeys',
                    ['global_person_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_journeys_global_person_id', table_name='journeys')
    with op.batch_alter_table('journeys') as batch_op:
        for col in ['global_person_id', 'camera_db_id', 'local_track_id',
                    'seen_at', 'confidence', 'confidence_caveat', 'created_at']:
            batch_op.drop_column(col)

    op.drop_index('ix_reid_review_items_status', table_name='reid_review_items')
    op.drop_table('reid_review_items')
    op.drop_table('global_persons')
    op.drop_table('reid_calibration_log')
