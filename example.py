import asyncio
import random
import time
from batcher import Batcher


def mock_model_inference(batch):
    time.sleep(0.01)
    return [f"Result for input: {item}" for item in batch]


async def mock_model_inference_async(batch):
    await asyncio.sleep(0.01)
    return [f"Async result for input: {item}" for item in batch]


async def simulate_client(batcher, client_id, num_requests=10):
    results = []
    for i in range(num_requests):
        data = f"Client{client_id}_Request{i}"
        result = await batcher.submit(data)
        results.append(result)
        await asyncio.sleep(random.uniform(0.001, 0.01))
    return results


async def benchmark_comparison():
    print("=" * 60)
    print("BENCHMARK: Batched vs Individual Processing")
    print("=" * 60)
    
    num_clients = 10
    requests_per_client = 20
    
    async def individual_inference(data):
        await asyncio.sleep(0.01)
        return f"Result: {data}"
    
    print(f"\n1. Individual Processing (no batching)")
    start_time = time.time()
    
    async def individual_client(client_id):
        results = []
        for i in range(requests_per_client):
            result = await individual_inference(f"Client{client_id}_Req{i}")
            results.append(result)
        return results
    
    await asyncio.gather(*[individual_client(i) for i in range(num_clients)])
    individual_time = time.time() - start_time
    print(f"   Total time: {individual_time:.2f}s")
    print(f"   Avg latency: {(individual_time / (num_clients * requests_per_client)) * 1000:.2f}ms")
    
    print(f"\n2. Batched Processing")
    batcher = Batcher(
        inference_fn=mock_model_inference_async,
        max_batch_size=32,
        max_wait_time=0.05
    )
    
    start_time = time.time()
    await asyncio.gather(*[
        simulate_client(batcher, i, requests_per_client) 
        for i in range(num_clients)
    ])
    batched_time = time.time() - start_time
    
    metrics = batcher.get_metrics()
    print(f"   Total time: {batched_time:.2f}s")
    print(f"   Avg latency: {metrics['average_latency_ms']:.2f}ms")
    print(f"   Total batches: {metrics['total_batches']}")
    print(f"   Avg batch size: {metrics['average_batch_size']:.2f}")
    
    improvement = ((individual_time - batched_time) / individual_time) * 100
    print(f"\n✅ Performance Improvement: {improvement:.1f}% reduction in total time")
    
    await batcher.stop()


async def main():
    print("=" * 60)
    print("BATCHER - Inference Microbatching Example")
    print("=" * 60)
    
    print("\n1. Basic Usage Example")
    print("-" * 40)
    
    batcher = Batcher(
        inference_fn=mock_model_inference,
        max_batch_size=32,
        max_wait_time=0.05
    )
    
    async def basic_example():
        tasks = []
        for i in range(5):
            task = batcher.submit(f"Request_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"   Request {i}: {result}")
    
    await basic_example()
    await batcher.stop()
    
    print("\n2. Concurrent Clients Example")
    print("-" * 40)
    
    batcher = Batcher(
        inference_fn=mock_model_inference_async,
        max_batch_size=16,
        max_wait_time=0.02
    )
    
    clients = await asyncio.gather(*[
        simulate_client(batcher, i, num_requests=5) 
        for i in range(3)
    ])
    
    print(f"   Processed {sum(len(c) for c in clients)} total requests")
    
    metrics = batcher.get_metrics()
    print("\n3. Metrics Report")
    print("-" * 40)
    print(f"   Total Requests: {metrics['total_requests']}")
    print(f"   Total Batches: {metrics['total_batches']}")
    print(f"   Requests/Second: {metrics['requests_per_second']}")
    print(f"   Avg Batch Size: {metrics['average_batch_size']}")
    print(f"   Avg Latency: {metrics['average_latency_ms']}ms")
    print(f"   Batch Sizes (last 10): {metrics['batch_size_distribution']['sizes']}")
    
    await batcher.stop()
    
    await benchmark_comparison()


if __name__ == "__main__":
    asyncio.run(main())