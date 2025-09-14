import asyncio
import time
from batcher import Batcher


def simple_inference(batch):
    return [f"processed_{item}" for item in batch]


async def run_tests():
    print("Running Batcher Tests...")
    print("=" * 40)
    
    print("\n✓ Test 1: Basic submission")
    batcher = Batcher(simple_inference, max_batch_size=5, max_wait_time=0.01)
    result = await batcher.submit("test_input")
    assert result == "processed_test_input", f"Expected 'processed_test_input', got {result}"
    await batcher.stop()
    print("  PASSED")
    
    print("\n✓ Test 2: Batch size limit")
    batch_sizes_seen = []
    
    def tracking_inference(batch):
        batch_sizes_seen.append(len(batch))
        return [f"result_{i}" for i in range(len(batch))]
    
    batcher = Batcher(tracking_inference, max_batch_size=3, max_wait_time=0.5)
    tasks = [batcher.submit(f"input_{i}") for i in range(10)]
    await asyncio.gather(*tasks)
    
    assert all(size <= 3 for size in batch_sizes_seen), f"Batch sizes exceeded limit: {batch_sizes_seen}"
    assert sum(batch_sizes_seen) == 10, f"Total items processed incorrect: {sum(batch_sizes_seen)}"
    await batcher.stop()
    print(f"  PASSED - Batch sizes: {batch_sizes_seen}")
    
    print("\n✓ Test 3: Timeout trigger")
    start_time = time.time()
    batcher = Batcher(simple_inference, max_batch_size=100, max_wait_time=0.05)
    await batcher.submit("single_request")
    elapsed = time.time() - start_time
    
    assert elapsed < 0.1, f"Timeout didn't trigger properly, took {elapsed}s"
    await batcher.stop()
    print(f"  PASSED - Request processed in {elapsed:.3f}s")
    
    print("\n✓ Test 4: Concurrent clients")
    batcher = Batcher(simple_inference, max_batch_size=10, max_wait_time=0.02)
    
    async def client(client_id):
        results = []
        for i in range(5):
            result = await batcher.submit(f"client_{client_id}_req_{i}")
            results.append(result)
        return results
    
    all_results = await asyncio.gather(*[client(i) for i in range(4)])
    total_results = sum(len(client_results) for client_results in all_results)
    assert total_results == 20, f"Expected 20 results, got {total_results}"
    
    metrics = batcher.get_metrics()
    assert metrics['total_requests'] == 20, f"Expected 20 requests, got {metrics['total_requests']}"
    await batcher.stop()
    print(f"  PASSED - Processed {total_results} requests in {metrics['total_batches']} batches")
    
    print("\n✓ Test 5: Order preservation")
    def ordered_inference(batch):
        return [f"result_{item}" for item in batch]
    
    batcher = Batcher(ordered_inference, max_batch_size=10, max_wait_time=0.02)
    inputs = [f"input_{i}" for i in range(20)]
    tasks = [batcher.submit(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)
    
    for i, (inp, result) in enumerate(zip(inputs, results)):
        assert result == f"result_{inp}", f"Order not preserved at index {i}"
    
    await batcher.stop()
    print("  PASSED - Order preserved for all 20 requests")
    
    print("\n✓ Test 6: Metrics collection")
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
    print(f"  PASSED - Metrics: RPS={metrics['requests_per_second']}, "
          f"Avg Batch={metrics['average_batch_size']}, "
          f"Avg Latency={metrics['average_latency_ms']}ms")
    
    print("\n" + "=" * 40)
    print("✅ All tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(run_tests())