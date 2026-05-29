"""Aurelius Hub — HuggingFace model upload support.

This package provides utilities for uploading trained models to
HuggingFace Hub from the Aurelius CLI.
"""

from __future__ import annotations

from aurelius.hub.uploader import upload_model_to_hub

__all__ = ["upload_model_to_hub"]
