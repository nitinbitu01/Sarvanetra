"""day15_push_notifications — push_subscriptions + push_delivery_log.

Hand-crafted for SQLite (same pattern as Day 5-14). Additive only.

push_delivery_log.subscription_id carries NO ForeignKey deliberately: an
expired subscription is deleted the moment a 410/404 is seen, and the audit
row must outlive it. A FK would either block the delete or cascade the log
row away — both destroy the record of an attempted notification.
"""
from alembic import op
import sqlalchemy as sa

revision = 'd15_push_notifications'
down_revision = 'd14_alert_routing'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'push_subscriptions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('officer_id', sa.Integer(), sa.ForeignKey('officers.id'),
                  nullable=False),
        sa.Column('endpoint', sa.Text(), nullable=False, unique=True),
        sa.Column('p256dh', sa.Text(), nullable=False),
        sa.Column('auth', sa.Text(), nullable=False),
        sa.Column('device_label', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.text("(datetime('now'))")),
        sa.Column('last_seen_at', sa.DateTime(), nullable=False,
                  server_default=sa.text("(datetime('now'))")),
    )
    op.create_index('idx_push_sub_officer', 'push_subscriptions', ['officer_id'])

    op.create_table(
        'push_delivery_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('alert_id', sa.Integer(), sa.ForeignKey('alerts.id'), nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=True),
        sa.Column('officer_id', sa.Integer(), sa.ForeignKey('officers.id'), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('error_detail', sa.Text(), nullable=True),
        sa.Column('sent_at', sa.DateTime(), nullable=False,
                  server_default=sa.text("(datetime('now'))")),
    )
    op.create_index('idx_push_log_alert', 'push_delivery_log', ['alert_id'])
    op.create_index('idx_push_log_officer', 'push_delivery_log', ['officer_id'])


def downgrade() -> None:
    op.drop_index('idx_push_log_officer', table_name='push_delivery_log')
    op.drop_index('idx_push_log_alert', table_name='push_delivery_log')
    op.drop_table('push_delivery_log')
    op.drop_index('idx_push_sub_officer', table_name='push_subscriptions')
    op.drop_table('push_subscriptions')
