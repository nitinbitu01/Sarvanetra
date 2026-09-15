"""add_journey_query_log_day9

Revision ID: 23d3352229b6
Revises: b7e2f91a3c5d
Create Date: 2026-08-20 01:40:46.303568

Day 9 §2: adds journey_query_log table for per-query RBAC audit trail.

NOTE: The autogenerate output included many spurious alter_column entries for
existing tables (SQLite reports ORM type annotations as schema differences but
does not support ALTER COLUMN). Those have been removed. Only the new table
creation is needed here — existing tables were created correctly by
Base.metadata.create_all() and prior migrations.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '23d3352229b6'
down_revision: Union[str, None] = 'b7e2f91a3c5d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'journey_query_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('role', sa.String(length=16), nullable=True),
        sa.Column('global_id_queried', sa.Integer(), nullable=True),
        sa.Column('query_time', sa.DateTime(), nullable=False),
        sa.Column('source_ip', sa.String(length=64), nullable=True),
        sa.Column('outcome', sa.String(length=16), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('journey_query_log')
