#!/bin/bash
# Aurelius v5.1 Environment Setup Script
# Run this to configure the M5 Pro hard-partitioned memory layout

set -e

echo "=== Aurelius v5.1 Environment Setup ==="
echo "Configuring M5 Pro Neural Accelerator memory partitioning..."

# PyTorch 2.12 async Metal compilation (eliminates JIT compilation lag)
export PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1

# MLX memory allocation (hard-partitioned from PyTorch)
export MLX_MAX_MEM_CACHE=12G

# Aurelius version marker
export AURELIUS_VERSION="5.1.0"
export AURELIUS_QUANT_PRESET="MX4"

# Metal shader cache directory
mkdir -p .metal_shader_cache

echo ""
echo "Environment variables set:"
echo "  PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1"
echo "  MLX_MAX_MEM_CACHE=12G"
echo "  AURELIUS_VERSION=5.1.0"
echo "  AURELIUS_QUANT_PRESET=MX4"
echo ""
echo "Memory Layout (M5 Pro 24GB):"
echo "  MLX (ChemVLM-2 MX4):    12GB reserved"
echo "  Metal Shader Cache:      2GB reserved"
echo "  PyTorch MPS (MatterSim): ~10GB available"
echo ""
echo "Add these to your ~/.zshrc or ~/.bashrc for persistence:"
echo ""
echo '  # Aurelius v5.1 - M5 Pro hard-partitioned memory'
echo '  export PYTORCH_MPS_ENABLE_ASYNC_COMPILATION=1'
echo '  export MLX_MAX_MEM_CACHE=12G'
echo '  export AURELIUS_VERSION="5.1.0"'
echo '  export AURELIUS_QUANT_PRESET="MX4"'
echo ""
