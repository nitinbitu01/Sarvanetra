"""day5_users_audit_soft_delete_evidence — Hand-crafted for SQLite compatibility.

SQLite cannot ALTER TABLE to add NOT NULL columns without defaults.
We add only truly new columns (nullable) and create new tables.
Type-change diffs on existing TEXT/REAL/TIMESTAMP columns are ignored
(SQLite is loosely typed — the ORM will handle them correctly at runtime).
"""
from alembic import op
import sqlalchemy as sa

revision = '3ae3fabcabe8'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── New table: users ─────────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('hashed_password', sa.String(length=256), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('badge_number', sa.String(length=32), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    # ── New table: audit_log ─────────────────────────────────────────────────
    op.create_table(
        'audit_log',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=64), nullable=False),
        sa.Column('resource_type', sa.String(length=32), nullable=True),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    # ── Add new nullable columns to cameras ──────────────────────────────────
    with op.batch_alter_table('cameras') as batch_op:
        batch_op.add_column(sa.Column('camera_id', sa.String(64), nullable=True))
        batch_op.add_column(sa.Column('ip_address', sa.String(64), nullable=True))
        batch_op.add_column(sa.Column('protocol', sa.String(16), nullable=True))
        batch_op.add_column(sa.Column('department', sa.String(64), nullable=True))
        batch_op.add_column(sa.Column('risk_level', sa.String(8), nullable=True))
        batch_op.add_column(sa.Column('added_by_user_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('deleted_at', sa.DateTime(), nullable=True))

    # ── Add new nullable columns to alerts ───────────────────────────────────
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.add_column(sa.Column('subject_label', sa.String(128), nullable=True))
        batch_op.add_column(sa.Column('confidence', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('danger_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('snapshot_path', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('evidence_hash', sa.String(64), nullable=True))
        batch_op.add_column(sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('deleted_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('alerts') as batch_op:
        for col in ['subject_label','confidence','danger_score','snapshot_path','evidence_hash','is_deleted','deleted_at']:
            batch_op.drop_column(col)
    with op.batch_alter_table('cameras') as batch_op:
        for col in ['camera_id','ip_address','protocol','department','risk_level','added_by_user_id','is_deleted','deleted_at']:
            batch_op.drop_column(col)
    op.drop_table('audit_log')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')
