"""day8_behavior_engine — camera_calibration table + alert lifecycle columns.

Hand-crafted for SQLite compatibility (same pattern as Day 5/6/7 migrations).
All new columns are nullable or carry a server_default; the new table is
additive only.

New table:
  - camera_calibration  (real-world px-per-meter per camera, §2)

Modified tables:
  - alerts  (adds lifecycle_status, reviewed_at, feedback — §3. Additive only;
             the existing status/false_positive_reason/reviewed_by_user_id
             columns from Day 5/7 are untouched, so /mark-false-positive and
             AlertFeed.jsx's false-positive UI keep working exactly as before.)
"""
from alembic import op
import sqlalchemy as sa

revision = 'b7e2f91a3c5d'
down_revision = 'f3a1c9d27b44'   # Day 7 migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'camera_calibration',
        sa.Column('camera_id', sa.String(length=64), primary_key=True),
        sa.Column('px_per_meter', sa.Float(), nullable=False),
        sa.Column('calibration_method', sa.String(length=32), nullable=False,
                  server_default='manual_two_point'),
        sa.Column('calibrated_at', sa.DateTime(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
    )

    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column('lifecycle_status', sa.String(16), nullable=False,
                                       server_default='OPEN'))
        batch_op.add_column(sa.Column('reviewed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('feedback', sa.String(16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.drop_column('feedback')
        batch_op.drop_column('reviewed_at')
        batch_op.drop_column('lifecycle_status')

    op.drop_table('camera_calibration')
