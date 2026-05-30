"""Pipeline configuration for Project Aurelius.

Provides default configuration values for the screening pipeline.
Memory management is delegated to PyTorch's native memory management.
"""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class AureliusConfig(BaseSettings):
    """Pipeline configuration for Project Aurelius.

    Provides default values for screening pipeline parameters.
    Does not manage hardware-specific memory allocation — that is
    delegated to PyTorch's native memory management.
    """

    weight_sigma: float = 0.4


def get_config() -> AureliusConfig:
    """Retrieve the configured AureliusPipeline settings.

    Returns:
        An AureliusConfig instance.
    """
    return AureliusConfig()


# Backward compatibility alias
AureliusConfig = AureliusConfig


def apply_global_config() -> AureliusConfig:
    """Apply configuration globally and return for use across modules.

    Returns:
        An AureliusConfig instance.
    """
    config = get_config()
    return config
