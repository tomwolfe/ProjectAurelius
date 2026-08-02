"""Aurelius feedback ingestion package.

Provides standardized tools for parsing experimental results from
CSV and SDF files, validating feedback schemas, and integrating
experimental data into the active-learning loop.
"""

from __future__ import annotations

from aurelius.feedback.parser import (
    ingest_feedback,
    parse_experimental_csv,
    parse_experimental_sdf,
    parse_feedback_file,
    validate_feedback_schema,
)

__all__ = [
    "ingest_feedback",
    "parse_experimental_csv",
    "parse_experimental_sdf",
    "parse_feedback_file",
    "validate_feedback_schema",
]
