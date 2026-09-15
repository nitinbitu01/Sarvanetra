-- migrations/init/00_pgvector_extension.sql
-- This script runs ONCE on first postgres container boot.
-- Creates the pgvector extension so the sentinel schema can use VECTOR columns.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- for fuzzy ANPR plate search
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- compound GIN indexes
