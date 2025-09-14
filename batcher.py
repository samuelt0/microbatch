import asyncio
import time
from typing import Any, Callable, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Request:
    data: Any
    future: asyncio.Future
    timestamp: float


class Batcher:
    def __init__(
        self,
        inference_fn: Callable,
        max_batch_size: int = 32,
        max_wait_time: float = 0.05
    ):
        self.inference_fn = inference_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self.queue = asyncio.Queue()
        self.worker_task = None
        self.is_running = False
        
        self.total_requests = 0
        self.total_batches = 0
        self.batch_sizes = []
        self.request_latencies = []
        self.start_time = time.time()
    
    async def start(self):
        if not self.is_running:
            self.is_running = True
            self.worker_task = asyncio.create_task(self._worker())
            logger.info(f"Batcher started with max_batch_size={self.max_batch_size}, max_wait_time={self.max_wait_time}s")
    
    async def stop(self):
        if self.is_running:
            self.is_running = False
            if self.worker_task:
                await self.worker_task
            logger.info("Batcher stopped")
    
    async def submit(self, request_data: Any) -> Any:
        if not self.is_running:
            await self.start()
        
        future = asyncio.Future()
        request = Request(
            data=request_data,
            future=future,
            timestamp=time.time()
        )
        
        await self.queue.put(request)
        result = await future
        
        latency = time.time() - request.timestamp
        self.request_latencies.append(latency)
        self.total_requests += 1
        
        return result
    
    async def _worker(self):
        while self.is_running:
            batch = []
            batch_start_time = time.time()
            
            try:
                while len(batch) < self.max_batch_size:
                    remaining_time = self.max_wait_time - (time.time() - batch_start_time)
                    
                    if remaining_time <= 0:
                        break
                    
                    try:
                        request = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=remaining_time
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
                for request in batch:
                    request.future.set_exception(e)
    
    async def _process_batch(self, batch: List[Request]):
        batch_size = len(batch)
        self.batch_sizes.append(batch_size)
        self.total_batches += 1
        
        batch_data = [request.data for request in batch]
        
        try:
            if asyncio.iscoroutinefunction(self.inference_fn):
                results = await self.inference_fn(batch_data)
            else:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.inference_fn, batch_data
                )
            
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError(f"Inference function must return a list of {len(batch)} results")
            
            for request, result in zip(batch, results):
                request.future.set_result(result)
                
            logger.debug(f"Processed batch of size {batch_size}")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for request in batch:
                request.future.set_exception(e)
    
    def get_metrics(self) -> dict:
        elapsed_time = time.time() - self.start_time
        rps = self.total_requests / elapsed_time if elapsed_time > 0 else 0
        
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        avg_latency = sum(self.request_latencies) / len(self.request_latencies) if self.request_latencies else 0
        
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "requests_per_second": round(rps, 2),
            "average_batch_size": round(avg_batch_size, 2),
            "average_latency_ms": round(avg_latency * 1000, 2),
            "batch_size_distribution": {
                "min": min(self.batch_sizes) if self.batch_sizes else 0,
                "max": max(self.batch_sizes) if self.batch_sizes else 0,
                "sizes": self.batch_sizes[-10:]
            }
        }