"""day14_alert_routing — officers + routed_alerts.

Hand-crafted for SQLite (same pattern as Day 5-13). Additive only: two new
tables, no existing table touched.

officers.current_alert_id carries NO ForeignKey on purpose — pairing it with
routed_alerts.assigned_officer → officers.id would create a circular FK that
SQLite cannot satisfy at CREATE TABLE time. The invariant is enforced by
conditional UPDATEs in backend/routing/service.py.
"""
from alembic import op
import sqlalchemy as sa

revision = 'd14_alert_routing'
down_revision = 'd13_evidence_lock'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'officers',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lng', sa.Float(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False,
                  server_default='AVAILABLE'),
        sa.Column('current_alert_id', sa.Integer(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=False,
                  server_default=sa.text("(datetime('now'))")),
    )

    op.create_table(
        'routed_alerts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('alert_id', sa.Integer(), sa.ForeignKey('alerts.id'), nullable=False),
        sa.Column('assigned_officer', sa.Integer(), sa.ForeignKey('officers.id'),
                  nullable=True),
        sa.Column('status', sa.String(length=24), nullable=False,
                  server_default='PENDING'),
        sa.Column('assigned_at', sa.DateTime(), nullable=True),
        sa.Column('ack_at', sa.DateTime(), nullable=True),
        sa.Column('escalated_at', sa.DateTime(), nullable=True),
        sa.Column('escalation_note', sa.Text(), nullable=True),
        sa.Column('escalation_count', sa.Integer(), nullable=False,
                  server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.text("(datetime('now'))")),
    )
    op.create_index('ix_routed_alerts_alert_id', 'routed_alerts', ['alert_id'])
    # The escalation tick scans on (status, escalation_count) every 5s.
    op.create_index('ix_routed_alerts_status_esc', 'routed_alerts',
                    ['status', 'escalation_count'])


def downgrade() -> None:
    op.drop_index('ix_routed_alerts_status_esc', table_name='routed_alerts')
    op.drop_index('ix_routed_alerts_alert_id', table_name='routed_alerts')
    op.drop_table('routed_alerts')
    op.drop_table('officers')
