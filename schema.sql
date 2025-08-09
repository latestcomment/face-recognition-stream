-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create sample table
CREATE TABLE person (
    person_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_name VARCHAR(255) NOT NULL,
    embedding vector(512) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);