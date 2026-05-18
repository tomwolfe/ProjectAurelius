#!/bin/bash
# Aurelius v6.0 Environment Setup Script
# Run this to configure the M5 Pro hard-partitioned memory layout
#
# These exports mirror the defaults in src/aurelius/config.py:
#   PYTORCH_MPS_ENABLE_ASYNC_COMPILATION -> M5ProConfig.pytorch_mps_async (default: True)
#   MLX_MAX_MEM_CACHE                  -> M5ProConfig.mlx_max_mem_gb (dynamic, 50% RAM capped at 12GB)
#   AURELIUS_VERSION                   -> hardcoded "6.0.0"
#   AURELIUS_QUANT_PRESET              -> M5ProConfig.chemvlm_quantization (default: "MX4")

set -e

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== Aurelius v6.0 Environment Setup (DRY RUN) ==="
    echo "The following exports would be set (matching config.py defaults):"
    echo ""
fi

echo "=== Aurelius v6.0 Environment Setup ==="
echo "Configuring M5 Pro Neural Accelerator memory partitioning..."

# PyTorch 2.12 async Metal compilation (eliminates JIT compilation lag)
# Mirrors: M5ProConfig.pytorch_mps_async = True
export PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1

# MLX memory allocation (hard-partitioned from PyTorch)
# Mirrors: M5ProConfig.mlx_max_mem_gb (dynamic: 50% of RAM, capped at 12GB)
# Note: For M5 Pro 24GB, this computes to 12GB
export MLX_MAX_MEM_CACHE=12G

# Aurelius version marker
# Mirrors: AURELIUS_VERSION in config.py apply_environment()
export AURELIUS_VERSION="6.0.0"
export AURELIUS_QUANT_PRESET="MX4"

# Metal shader cache directory
mkdir -p .metal_shader_cache

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Dry run - exports that would be applied:"
    echo "  export PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1"
    echo "  export MLX_MAX_MEM_CACHE=12G"
    echo "  export AURELIUS_VERSION=\"6.0.0\""
    echo "  export AURELIUS_QUANT_PRESET=\"MX4\""
    echo ""
    echo "To apply, run without --dry-run:"
    echo "  source setup_env.sh"
    echo ""
else
    echo ""
    echo "Environment variables set:"
    echo "  PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1"
    echo "  MLX_MAX_MEM_CACHE=12G"
    echo "  AURELIUS_VERSION=6.0.0"
    echo "  AURELIUS_QUANT_PRESET=MX4"
    echo ""
    echo "Memory Layout (M5 Pro 24GB):"
    echo "  MLX (ChemVLM-2 MX4):    12GB reserved"
    echo "  Metal Shader Cache:      2GB reserved"
    echo "  PyTorch MPS (MatterSim): ~10GB available"
    echo ""
    echo "Add these to your ~/.zshrc or ~/.bashrc for persistence:"
    echo ""
    echo '  # Aurelius v6.0 - M5 Pro hard-partitioned memory'
    echo '  export PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1'
    echo '  export MLX_MAX_MEM_CACHE=12G'
    echo '  export AURELIUS_VERSION="6.0.0"'
    echo '  export AURELIUS_QUANT_PRESET="MX4"'
    echo ""
fi
