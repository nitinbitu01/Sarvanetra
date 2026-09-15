"""day17_watchlist_plate_active — active flag on watchlist_plates.

Hand-crafted for SQLite compatibility (same pattern as Day 5-16). Additive
only — one nullable-with-default column.

Part of unifying vehicle-plate watchlist matching onto watchlist_plates as
the single source of truth (see docs/WATCHLIST_ALERTING_ARCHITECTURE.md):
the live ANPR pipeline previously matched against a standalone watchlist.json
file that nothing in the app's UI or API could edit, while plate-search and
journey lookups already read watchlist_plates — two watchlists, only one of
them reachable by an operator. `active` lets an entry be retired (case
resolved) without deleting the row a past Alert may reference by plate text.
"""
from alembic import op
import sqlalchemy as sa

revision = 'd17_watchlist_plate_active'
down_revision = 'd16_camera_registry_enrichment'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('watchlist_plates') as batch_op:
        batch_op.add_column(sa.Column('active', sa.Boolean(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('watchlist_plates') as batch_op:
        batch_op.drop_column('active')
