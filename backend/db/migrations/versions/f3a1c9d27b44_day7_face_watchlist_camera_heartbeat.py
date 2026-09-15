"""day7_face_watchlist_camera_heartbeat — Alert false-positive review + Camera heartbeat columns.

Hand-crafted for SQLite compatibility (same pattern as Day 5/6 migrations).
All new columns are nullable or carry a server_default.

watchlist_persons is NOT touched here — Day 0/3 setup already created it with
face_embedding (LargeBinary, not null) + embedding_dim (Integer, not null),
which is exactly what Day 7's face matcher needs. Verified at runtime by
face_watchlist_matcher.reload_watchlist() (skips + logs any row whose stored
embedding_dim doesn't match FACE_EMBEDDING_DIM instead of failing silently).

Modified tables:
  - alerts   (adds false_positive_reason, reviewed_by_user_id)
  - cameras  (adds last_heartbeat_at, consecutive_failures)
"""
from alembic import op
import sqlalchemy as sa

revision = 'f3a1c9d27b44'
down_revision = 'a1b2c3d4e5f6'   # Day 6 migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column('false_positive_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('reviewed_by_user_id', sa.Integer(), nullable=True))

    with op.batch_alter_table('cameras') as batch_op:
        batch_op.add_column(sa.Column('last_heartbeat_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('consecutive_failures', sa.Integer(), nullable=False,
                                       server_default='0'))


def downgrade() -> None:
    with op.batch_alter_table('cameras') as batch_op:
        batch_op.drop_column('consecutive_failures')
        batch_op.drop_column('last_heartbeat_at')

    with op.batch_alter_table('alerts') as batch_op:
        batch_op.drop_column('reviewed_by_user_id')
        batch_op.drop_column('false_positive_reason')
