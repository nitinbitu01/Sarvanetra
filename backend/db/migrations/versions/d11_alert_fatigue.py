"""day11_alert_fatigue — dedup + zone-incident schema.

Additive-only, SQLite-safe (batch_alter_table). Two changes:
  1. alerts.merged_into_alert_id (nullable Integer FK → alerts.id)
     Set on the losing alert when apply_dedup() picks a primary winner.
     NULL = this alert is a primary (was never merged away).
     Non-null = this alert is a secondary; merged_into_alert_id points at
     the primary. The row is NEVER deleted — evidence integrity.
  2. zone_incidents table (new) — lightweight aggregate row created when
     5+ Alert rows fire within 100m (haversine) in 2 min. Constituent
     alert PKs are stored as a JSON list in alert_ids (Text), never hidden.

down_revision: c4d8e1f60a92 (Day 10 sentinel_iq migration)
"""
from alembic import op
import sqlalchemy as sa

revision = 'd11_alert_fatigue'
down_revision = 'c4d8e1f60a92'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add merged_into_alert_id to alerts
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column(
            'merged_into_alert_id',
            sa.Integer(),
            sa.ForeignKey('alerts.id', name='fk_alerts_merged_into_alert_id'),
            nullable=True,
        ))

    # 2. Create zone_incidents table
    op.create_table(
        'zone_incidents',
        sa.Column('id',         sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('zone',       sa.String(64), nullable=False),
        sa.Column('center_lat', sa.Float(),    nullable=False),
        sa.Column('center_lon', sa.Float(),    nullable=False),
        sa.Column('opened_at',  sa.DateTime(), nullable=False,
                  server_default=sa.func.current_timestamp()),
        sa.Column('closed_at',  sa.DateTime(), nullable=True),
        # JSON list of Alert.id — constituent alerts, never hidden
        sa.Column('alert_ids',  sa.Text(),     nullable=False),
    )


def downgrade() -> None:
    op.drop_table('zone_incidents')

    with op.batch_alter_table('alerts') as batch_op:
        batch_op.drop_column('merged_into_alert_id')
