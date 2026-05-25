"""Data resources package for Project Aurelius.

This package provides fallback data files for training and screening
when external datasets are unavailable (e.g., no network, missing
HuggingFace access).

Usage:
    from importlib import resources
    esol = resources.files("aurelius.data").joinpath("esol_fallback.csv")
"""
