import asyncio
import time
import numpy as np
from batcher import Batcher

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")


class SimpleNet(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def create_pytorch_batcher():
    """Create a batcher for PyTorch model inference"""
    
    # Initialize model
    model = SimpleNet()
    model.eval()  # Set to evaluation mode
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    def pytorch_batch_inference(batch):
        """Process batch of numpy arrays through PyTorch model"""
        with torch.no_grad():
            # Convert list of numpy arrays to tensor batch
            inputs = torch.FloatTensor(np.vstack(batch)).to(device)
            
            # Single forward pass for entire batch
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            
            # Convert back to Python lists
            results = []
            for i in range(len(batch)):
                results.append({
                    'class': int(predicted[i]),
                    'confidence': float(probs[i].max()),
                    'probabilities': probs[i].cpu().numpy().tolist()
                })
            
            return results
    
    return pytorch_batch_inference, model, device


async def benchmark_pytorch():
    """Benchmark PyTorch inference with and without batching"""
    
    if not PYTORCH_AVAILABLE:
        print("Skipping PyTorch benchmark - PyTorch not installed")
        return
    
    print("=" * 60)
    print("PYTORCH NEURAL NETWORK BATCHING")
    print("=" * 60)
    
    inference_fn, model, device = create_pytorch_batcher()
    
    # Create batcher
    batcher = Batcher(
        inference_fn=inference_fn,
        max_batch_size=64,
        max_wait_time=0.02  # 20ms
    )
    
    # Generate test data (MNIST-like 28x28 images flattened)
    num_samples = 500
    test_data = [np.random.randn(784).astype(np.float32) for _ in range(num_samples)]
    
    print(f"\nBenchmarking with {num_samples} samples...")
    
    # Benchmark individual inference
    print("\n1. Individual Inference (one sample at a time):")
    start = time.time()
    individual_results = []
    with torch.no_grad():
        for data in test_data:
            input_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            individual_results.append(int(predicted[0]))
    individual_time = time.time() - start
    
    print(f"   Time: {individual_time:.3f}s")
    print(f"   Throughput: {num_samples/individual_time:.1f} samples/sec")
    print(f"   Avg latency: {individual_time/num_samples*1000:.2f}ms per sample")
    
    # Benchmark batched inference
    print("\n2. Batched Inference (using Batcher):")
    start = time.time()
    tasks = [batcher.submit(data) for data in test_data]
    batched_results = await asyncio.gather(*tasks)
    batched_time = time.time() - start
    
    metrics = batcher.get_metrics()
    
    print(f"   Time: {batched_time:.3f}s")
    print(f"   Throughput: {num_samples/batched_time:.1f} samples/sec")
    print(f"   Avg latency: {metrics['average_latency_ms']:.2f}ms per sample")
    print(f"   Avg batch size: {metrics['average_batch_size']:.1f}")
    print(f"   Total batches: {metrics['total_batches']}")
    
    speedup = individual_time / batched_time
    print(f"\n✅ Speedup: {speedup:.2f}x faster with batching!")
    
    await batcher.stop()
    
    # Simulate concurrent clients
    print("\n" + "=" * 60)
    print("SIMULATING CONCURRENT CLIENTS")
    print("=" * 60)
    
    batcher = Batcher(
        inference_fn=inference_fn,
        max_batch_size=32,
        max_wait_time=0.015
    )
    
    async def client(client_id, num_requests=20):
        """Simulate a client making sequential requests"""
        results = []
        for i in range(num_requests):
            # Random delay between requests
            await asyncio.sleep(np.random.uniform(0.001, 0.005))
            
            # Submit request
            data = np.random.randn(784).astype(np.float32)
            result = await batcher.submit(data)
            results.append(result)
        
        return results
    
    # Run 10 concurrent clients
    num_clients = 10
    print(f"\nRunning {num_clients} concurrent clients...")
    
    start = time.time()
    client_tasks = [client(i) for i in range(num_clients)]
    all_results = await asyncio.gather(*client_tasks)
    concurrent_time = time.time() - start
    
    total_requests = sum(len(r) for r in all_results)
    metrics = batcher.get_metrics()
    
    print(f"\nResults:")
    print(f"   Total requests: {total_requests}")
    print(f"   Total time: {concurrent_time:.3f}s")
    print(f"   Overall throughput: {total_requests/concurrent_time:.1f} req/sec")
    print(f"   Average latency: {metrics['average_latency_ms']:.2f}ms")
    print(f"   Batch efficiency: {metrics['average_batch_size']:.1f} samples/batch")
    print(f"   Batch size range: {metrics['batch_size_distribution']['min']}-{metrics['batch_size_distribution']['max']}")
    
    await batcher.stop()


async def advanced_pytorch_example():
    """Advanced example with different model types"""
    
    if not PYTORCH_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("ADVANCED: DIFFERENT MODEL ARCHITECTURES")
    print("=" * 60)
    
    # CNN for image classification
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    cnn_model = SimpleCNN()
    cnn_model.eval()
    
    def cnn_batch_inference(batch):
        """Process batch of images through CNN"""
        with torch.no_grad():
            # batch is list of (3, 32, 32) images
            inputs = torch.FloatTensor(np.stack(batch))
            outputs = cnn_model(inputs)
            _, predicted = torch.max(outputs, 1)
            return predicted.tolist()
    
    # Create batcher for CNN
    cnn_batcher = Batcher(
        inference_fn=cnn_batch_inference,
        max_batch_size=128,  # CNNs can handle larger batches
        max_wait_time=0.05
    )
    
    print("\nCNN Model Batching:")
    # Generate fake image data (3, 32, 32)
    images = [np.random.randn(3, 32, 32).astype(np.float32) for _ in range(100)]
    
    start = time.time()
    tasks = [cnn_batcher.submit(img) for img in images]
    results = await asyncio.gather(*tasks)
    cnn_time = time.time() - start
    
    metrics = cnn_batcher.get_metrics()
    print(f"   Processed {len(images)} images in {cnn_time:.3f}s")
    print(f"   Throughput: {len(images)/cnn_time:.1f} images/sec")
    print(f"   Average batch size: {metrics['average_batch_size']:.1f}")
    
    await cnn_batcher.stop()


async def main():
    if PYTORCH_AVAILABLE:
        await benchmark_pytorch()
        await advanced_pytorch_example()
        
        print("\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)
        print("✅ Batching dramatically improves throughput (2-5x typical)")
        print("✅ GPU utilization increases with batch size")
        print("✅ Latency remains controlled via max_wait_time")
        print("✅ Works seamlessly with PyTorch models")
    else:
        print("\n" + "=" * 60)
        print("To run PyTorch examples, install PyTorch:")
        print("  pip install torch")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())