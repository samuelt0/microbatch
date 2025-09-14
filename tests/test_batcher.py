import asyncio
import pytest
import time
from batcher import Batcher


def simple_inference(batch):
    return [f"processed_{item}" for item in batch]


async def async_inference(batch):
    await asyncio.sleep(0.001)
    return [f"async_processed_{item}" for item in batch]


def slow_inference(batch):
    time.sleep(0.1)
    return [item * 2 for item in batch]


@pytest.mark.asyncio
async def test_basic_submission():
    batcher = Batcher(simple_inference, max_batch_size=5, max_wait_time=0.01)
    
    result = await batcher.submit("test_input")
    assert result == "processed_test_input"
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_size_limit():
    batch_sizes_seen = []
    
    def tracking_inference(batch):
        batch_sizes_seen.append(len(batch))
        return [f"result_{i}" for i in range(len(batch))]
    
    batcher = Batcher(tracking_inference, max_batch_size=3, max_wait_time=0.5)
    
    tasks = [batcher.submit(f"input_{i}") for i in range(10)]
    await asyncio.gather(*tasks)
    
    assert all(size <= 3 for size in batch_sizes_seen)
    assert sum(batch_sizes_seen) == 10
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_timeout_trigger():
    call_times = []
    
    def time_tracking_inference(batch):
        call_times.append(time.time())
        return [f"result_{item}" for item in batch]
    
    batcher = Batcher(time_tracking_inference, max_batch_size=100, max_wait_time=0.05)
    
    await batcher.submit("single_request")
    
    assert len(call_times) == 1
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_async_inference_function():
    batcher = Batcher(async_inference, max_batch_size=5, max_wait_time=0.01)
    
    tasks = [batcher.submit(f"input_{i}") for i in range(3)]
    results = await asyncio.gather(*tasks)
    
    assert results == ["async_processed_input_0", "async_processed_input_1", "async_processed_input_2"]
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_concurrent_clients():
    batcher = Batcher(simple_inference, max_batch_size=10, max_wait_time=0.02)
    
    async def client(client_id):
        results = []
        for i in range(5):
            result = await batcher.submit(f"client_{client_id}_req_{i}")
            results.append(result)
        return results
    
    all_results = await asyncio.gather(*[client(i) for i in range(4)])
    
    total_results = sum(len(client_results) for client_results in all_results)
    assert total_results == 20
    
    metrics = batcher.get_metrics()
    assert metrics['total_requests'] == 20
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_metrics_collection():
    batcher = Batcher(simple_inference, max_batch_size=5, max_wait_time=0.01)
    
    for i in range(10):
        await batcher.submit(f"request_{i}")
    
    metrics = batcher.get_metrics()
    
    assert metrics['total_requests'] == 10
    assert metrics['total_batches'] >= 2
    assert metrics['requests_per_second'] > 0
    assert metrics['average_batch_size'] > 0
    assert metrics['average_latency_ms'] > 0
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_error_handling():
    def failing_inference(batch):
        if len(batch) > 2:
            raise ValueError("Batch too large!")
        return [f"ok_{item}" for item in batch]
    
    batcher = Batcher(failing_inference, max_batch_size=5, max_wait_time=0.01)
    
    tasks = [batcher.submit(f"input_{i}") for i in range(5)]
    
    with pytest.raises(ValueError):
        await asyncio.gather(*tasks)
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_order_preservation():
    def ordered_inference(batch):
        return [f"result_{item}" for item in batch]
    
    batcher = Batcher(ordered_inference, max_batch_size=10, max_wait_time=0.02)
    
    inputs = [f"input_{i}" for i in range(20)]
    tasks = [batcher.submit(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)
    
    for i, (inp, result) in enumerate(zip(inputs, results)):
        assert result == f"result_{inp}"
    
    await batcher.stop()


@pytest.mark.asyncio
async def test_auto_start():
    batcher = Batcher(simple_inference, max_batch_size=5, max_wait_time=0.01)
    
    assert not batcher.is_running
    
    await batcher.submit("test")
    
    assert batcher.is_running
    
    await batcher.stop()
    assert not batcher.is_running


if __name__ == "__main__":
    asyncio.run(test_basic_submission())
    asyncio.run(test_batch_size_limit())
    asyncio.run(test_timeout_trigger())
    asyncio.run(test_async_inference_function())
    asyncio.run(test_concurrent_clients())
    asyncio.run(test_metrics_collection())
    asyncio.run(test_order_preservation())
    asyncio.run(test_auto_start())
    print("✅ All tests passed!")