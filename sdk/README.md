# Aurelius SDK — `aurelius-sdk`

Thin Python client for the [Project Aurelius](https://github.com/anomalyco/ProjectAurelius) discovery engine API.

## Installation

```bash
pip install aurelius-sdk
```

Requires `httpx` (installed automatically as a dependency).

## Quick Start

```python
from aurelius_sdk import Client

# 1. Initialize the client
client = Client(base_url="http://localhost:8000", api_key="your-api-key")

# 2. Screen a molecule
result = client.screen("CCO")
print(result["homo_eV"])         # -7.2
print(result["lumo_eV"])         #  0.8
print(result["dielectric_proxy"]) # 24.5

# 3. Batch screen multiple molecules
results = client.screen_batch(["CCO", "CCCO", "C1CCOC1"])
for r in results:
    print(r["smiles"], r["homo_eV"])

# 4. Check server health
health = client.health()
print(health["status"])  # "ok"

# Use as a context manager
with Client(base_url="http://localhost:8000") as c:
    data = c.screen("C1COC(=O)O1")
    print(data)
```

## Loading a Pre-Certified Kernel

Pre-certified kernels (see [`docs/examples/kernels/`](../docs/examples/kernels/))
are JSON files produced by the Certification Lab. They are designed for local
engine use — load them directly into your analysis pipeline:

```python
import json

with open("docs/examples/kernels/carbonate_high_voltage.json") as f:
    kernel = json.load(f)

# Inspect tuned parameters
print(kernel["tom_parameters"])
# {'homo_offset': 0.0, 'lumo_offset': 0.0, 'gc_scale': 4.5, 'uq_scale': 0.95}

# Check validation metrics
print(kernel["validation_metrics"])
```

> **Note:** Kernel loading is a client-side operation. The API does not expose
> a kernel-loading endpoint — pre-certified kernels are meant to be consumed
> by the engine's local screening pipeline.

## API

### `Client(base_url, api_key, timeout)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:8000` | Base URL of the Aurelius engine API |
| `api_key` | `None` | API key (sent as `X-API-Key` header) |
| `timeout` | `30.0` | Request timeout in seconds |

### Methods

| Method | Description |
|--------|-------------|
| `screen(smiles)` | Screen a single molecule, return full evaluation |
| `screen_batch(smiles_list)` | Screen multiple molecules in one request |
| `health()` | Check server health status |
| `close()` | Close the underlying HTTP client |

All methods raise `httpx.HTTPStatusError` on non-2xx responses.

## Development

```bash
cd sdk
pip install -e .
pytest tests/ -v
```

## License

MIT — see [`engine/LICENSE`](../engine/LICENSE).
