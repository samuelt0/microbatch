# Batcher - Lightweight Inference Microbatching

A minimal Python library for batching requests for ML inference

## Quick Start

```python
from batcher import Batcher

# Define your inference function
def model_inference(batch):
    # Process batch of inputs
    return [model.predict(item) for item in batch]

# Create batcher
batcher = Batcher(
    inference_fn=model_inference,
    max_batch_size=32,
    max_wait_time=0.05  # 50ms
)

# Submit requests (automatically batched)
result = await batcher.submit(input_data)
```

## Features

- **Automatic Batching** - Groups requests up to `max_batch_size`
- **Timeout Control** - Flushes incomplete batches after `max_wait_time`
- **Async Support** - Built on asyncio for high concurrency
- **Order Preservation** - Results returned in submission order
- **Built-in Metrics** - Track RPS, latency, and batch sizes

## Installation

```bash
pip install -r requirements.txt
```

## Examples

Run the example to see batching in action:

```bash
python example.py
```

Run tests:

```bash
python simple_test.py
```

## Metrics

```python
metrics = batcher.get_metrics()
# {
#   'requests_per_second': 156.3,
#   'average_batch_size': 24.5,
#   'average_latency_ms': 32.1,
#   ...
# }
```
