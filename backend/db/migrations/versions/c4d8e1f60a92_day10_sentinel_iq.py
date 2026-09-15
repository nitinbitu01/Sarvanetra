"""day10_sentinel_iq — per-alert SENTINEL IQ contribution + breakdown.

Hand-crafted for SQLite compatibility (same pattern as the Day 5/6/7/8
migrations). Additive only — two nullable columns on `alerts`.

Why two new columns rather than reusing `danger_score`:
  Different scale (single-digit IQ vs 0-100 danger_score), different history
  (Day 7 wrote a fixed 95.0 into danger_score, Day 8 left it None), and the
  frontend's existing danger_score badge thresholds are calibrated to the
  0-100 range. See the Alert model docstring in backend/db/models.py.

NULL semantics for iq_contribution:
  NULL = "this alert_type has no base score" (roadmap-only trigger). It is
  NOT the same as 0.0, which would be a genuinely computed zero. The score
  card and the aggregate query both treat NULL as "skip", never as "add 0".
"""
from alembic import op
import sqlalchemy as sa

revision = 'c4d8e1f60a92'
down_revision = '23d3352229b6'   # Day 9 journey_query_log migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column('iq_contribution', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('iq_breakdown_json', sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.drop_column('iq_breakdown_json')
        batch_op.drop_column('iq_contribution')
