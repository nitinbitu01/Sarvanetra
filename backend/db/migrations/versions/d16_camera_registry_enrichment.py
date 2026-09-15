"""day16_camera_registry_enrichment — camera_type, installed_at, maintenance
fields on `cameras`.

Hand-crafted for SQLite compatibility (same pattern as Day 5-15). Additive
only — five nullable columns on `cameras`, no data migration needed since
every existing row simply reads NULL/default for fields that did not exist
when it was created.

Closes three Model 1 gaps found by an evidence-based checklist review against
the official deliverable list (docs/MODEL_1_ARCHITECTURE.md):
  - "camera type" GIS map layer had no backing column at all
  - "ageing infrastructure" gap-analysis had no install-date to age against
  - "maintenance-status monitoring" only had automatic ONLINE/OFFLINE from
    camera_heartbeat.py, no manually-set maintenance state
"""
from alembic import op
import sqlalchemy as sa

revision = 'd16_camera_registry_enrichment'
down_revision = 'd15_push_notifications'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('cameras') as batch_op:
        batch_op.add_column(sa.Column('camera_type', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('installed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('maintenance_mode', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('maintenance_note', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('maintenance_since', sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('cameras') as batch_op:
        batch_op.drop_column('maintenance_since')
        batch_op.drop_column('maintenance_note')
        batch_op.drop_column('maintenance_mode')
        batch_op.drop_column('installed_at')
        batch_op.drop_column('camera_type')
