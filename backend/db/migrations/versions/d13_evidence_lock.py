"""day13_evidence_lock — evidence table (auto clip + chain of custody).

Hand-crafted for SQLite compatibility (same pattern as Day 5-11 migrations).
Additive only: one new table, nothing existing is touched.

The UNIQUE constraint on alert_id is load-bearing, not decorative — it is
what makes evidence capture idempotent under concurrency. See the Evidence
model docstring in backend/db/models.py.
"""
from alembic import op
import sqlalchemy as sa

revision = 'd13_evidence_lock'
down_revision = 'd11_alert_fatigue'   # Day 11 migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'evidence',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('alert_id', sa.Integer(), sa.ForeignKey('alerts.id'),
                  nullable=False, unique=True),
        sa.Column('camera_db_id', sa.Integer(), sa.ForeignKey('cameras.id'), nullable=True),
        sa.Column('camera_str_id', sa.String(length=64), nullable=True),

        sa.Column('status', sa.String(length=16), nullable=False, server_default='PENDING'),
        sa.Column('failure_reason', sa.Text(), nullable=True),

        sa.Column('clip_path', sa.Text(), nullable=True),
        sa.Column('pdf_path', sa.Text(), nullable=True),
        sa.Column('sha256', sa.String(length=64), nullable=True),

        sa.Column('event_time', sa.DateTime(), nullable=False),
        sa.Column('clip_start_time', sa.DateTime(), nullable=True),
        sa.Column('clip_end_time', sa.DateTime(), nullable=True),

        sa.Column('requested_pre_seconds', sa.Float(), nullable=True),
        sa.Column('requested_post_seconds', sa.Float(), nullable=True),
        sa.Column('actual_pre_seconds', sa.Float(), nullable=True),
        sa.Column('actual_post_seconds', sa.Float(), nullable=True),
        sa.Column('actual_duration_seconds', sa.Float(), nullable=True),

        sa.Column('frame_count', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('codec', sa.String(length=8), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),

        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_evidence_alert_id', 'evidence', ['alert_id'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_evidence_alert_id', table_name='evidence')
    op.drop_table('evidence')
