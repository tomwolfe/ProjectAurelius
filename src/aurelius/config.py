"""Pipeline configuration for Project Aurelius.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AureliusConfig:
    """Pipeline configuration for Project Aurelius.
    """


def get_config() -> AureliusConfig:
    """Retrieve the configured AureliusPipeline settings.

    Returns:
        An AureliusConfig instance.
    """
    return AureliusConfig()


def apply_global_config() -> AureliusConfig:
    """Apply configuration globally and return for use across modules.

    Returns:
        An AureliusConfig instance.
    """
    config = get_config()
    return config
