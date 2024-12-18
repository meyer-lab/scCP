from functools import wraps
import threading
import time
import GPUtil
import cupy as cp
import numpy as np
import torch

class GPUMaxMemoryTracker:
    """Tracks maximum GPU memory usage during function execution."""
    
    def __init__(self, device_id=0, sampling_rate=0.1):
        self.device_id = device_id
        self.sampling_rate = sampling_rate
        self.max_memory = 0
        self._stop_monitoring = False
        
    def _monitor_memory(self):
        """Continuously monitors GPU memory usage."""
        pool = cp.get_default_memory_pool()
        while not self._stop_monitoring:
            try:
                gpu = GPUtil.getGPUs()[self.device_id]
                self.max_memory = max(self.max_memory, gpu.memoryUsed)
                time.sleep(self.sampling_rate)
            except Exception as e:
                print(f"Error monitoring GPU memory: {e}")
                break
    
    def track(self, func):
        """Decorator to track maximum memory usage during function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.max_memory = 0
            self._stop_monitoring = False
            
            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(target=self._monitor_memory)
            monitor_thread.start()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop monitoring and wait for thread to finish
                self._stop_monitoring = True
                monitor_thread.join()
        
        return wrapper
    
    def get_max_memory(self):
        return self.max_memory


x = cp.random.randn(1000)
print(cp.get_default_memory_pool().total_bytes())
print(torch.cuda.memory_allocated())
del x
cp.get_default_memory_pool().free_all_blocks()
print(cp.get_default_memory_pool().total_bytes())
print(torch.cuda.memory_allocated())

y = torch.randn(1000, device='cuda')
print(cp.get_default_memory_pool().total_bytes())
print(torch.cuda.memory_allocated())
del y
print(cp.get_default_memory_pool().total_bytes())
print(torch.cuda.memory_allocated())
