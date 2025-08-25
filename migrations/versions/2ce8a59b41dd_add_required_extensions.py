"""Add required extensions

Revision ID: 2ce8a59b41dd
Revises: d6d3adb1cedf
Create Date: 2025-08-25 14:42:10.763963

"""
from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision = '2ce8a59b41dd'
down_revision = 'd6d3adb1cedf'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create required PostgreSQL extensions  
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def downgrade() -> None:
    # Drop extensions (only if no objects depend on them)
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")