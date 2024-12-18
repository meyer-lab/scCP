# %%
import anndata
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time
import tracemalloc
from typing import List, Dict
from pathlib import Path
import seaborn as sns
import pynvml  # For GPU memory monitoring
import sccp.factorization
from importlib import reload
reload(sccp.factorization)
from memory_profiler import profile

from harmonypy import run_harmony

from sccp.factorization import pf2
from sccp.imports import import_lupus
from sccp.figures.common import getSetup, subplotLabel
import gc


RECOMPUTE = True  # Set to True to rerun benchmarks, False to load from saved results

DATA_DIR = Path(__file__).parent.parent / "data"

def makeFigure():
    """Create figure comparing computational requirements of pf2 and Harmony."""
    ax, f = getSetup((15, 4), (1, 3))  # Adjusted to accommodate an additional subplot

    # Load or compute benchmark results
    if RECOMPUTE:
        cell_counts = [100]
        run_benchmarks(cell_counts)

    # Load results
    results_path = DATA_DIR / "benchmark_results.csv"
    df_results = pd.read_csv(results_path)

    # Plot runtime comparison
    sns.lineplot(
        data=df_results,
        x="cell_count",
        y="avg_runtime",
        hue="algorithm",
        ax=ax[0],
        marker='o'
    )
    ax[0].set_xlabel("Number of Cells")
    ax[0].set_ylabel("Runtime (seconds)")
    ax[0].set_title("Runtime Comparison")

    # Plot CPU memory usage comparison
    sns.lineplot(
        data=df_results,
        x="cell_count",
        y="avg_cpu_memory",
        hue="algorithm",
        ax=ax[1],
        marker='o'
    )
    ax[1].set_xlabel("Number of Cells")
    ax[1].set_ylabel("Max CPU Memory Usage (MB)")
    ax[1].set_title("CPU Memory Usage Comparison")

    # Plot GPU memory usage comparison
    sns.lineplot(
        data=df_results,
        x="cell_count",
        y="avg_gpu_memory",
        hue="algorithm",
        ax=ax[2],
        marker='o'
    )
    ax[2].set_xlabel("Number of Cells")
    ax[2].set_ylabel("Max GPU Memory Usage (MB)")
    ax[2].set_title("GPU Memory Usage Comparison")

    # Add subplot labels
    subplotLabel(ax)

    return f

import GPUtil
from functools import wraps
import threading
import time

class GPUMaxMemoryTracker:
    """Tracks maximum GPU memory usage during function execution."""
    
    def __init__(self, device_id=0, sampling_rate=0.1):
        self.device_id = device_id
        self.sampling_rate = sampling_rate
        self.max_memory = 0
        self._stop_monitoring = False
        
    def _monitor_memory(self):
        """Continuously monitors GPU memory usage."""
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
        """Returns the maximum memory usage in MB."""
        return self.max_memory

def benchmark_algorithm(
    data: anndata.AnnData,
    algorithm: str,
    rank: int = 20,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmarks the specified algorithm on the given data.

    Parameters:
        data: anndata.AnnData object containing the data.
        algorithm: 'pf2' or 'Harmony'.
        rank: The number of components for factorization.
        **kwargs: Additional keyword arguments.

    Returns:
        A dictionary with runtime, max CPU memory usage, and max GPU memory usage.
    """
    # Initialize GPU memory tracking
    tracker = GPUMaxMemoryTracker(device_id=0)

    start_time = time.time()
    tracemalloc.start()

    if algorithm == 'pf2':
        # Run pf2 algorithm
        assert data.shape[0] > rank, "Number of cells must be greater than rank"

        @tracker.track
        def run():
            pf2(data, rank=rank, doEmbedding=False, max_iter=2, **kwargs)
        run()

    elif algorithm == 'Harmony':
        # Run Harmony algorithm using harmonypy

        # Ensure data.X is a dense array
        X = data.X.toarray() if hasattr(data.X, 'toarray') else data.X

        # Prepare metadata
        if 'pool' not in data.obs:
            raise ValueError("Data must have 'batch' column in .obs for Harmony integration.")
        meta_data = data.obs[['pool']]

        # Run Harmony with GPU acceleration (if supported)
        @tracker.track
        def run():
            run_harmony(X, meta_data, vars_use=['pool'])
        run()

    else:
        raise ValueError("Algorithm must be 'pf2' or 'Harmony'")

    # Measure CPU memory usage
    current, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    # Measure GPU memory usage
    max_gpu_memory = tracker.get_max_memory()

    runtime = end_time - start_time
    max_cpu_memory = peak_cpu / 1e6  # Convert bytes to megabytes

    return {
        'runtime': runtime,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory
    }

# %%

"""
Runs benchmarks for different cell counts and saves the results.

Parameters:
    cell_counts: List of cell counts to test.
    n_runs: Number of runs for averaging.
"""

X = import_lupus()

# %%
import gc
import sys

import torch


def super_aggressive_cuda_cleanup():
    """
    Much more aggressive CUDA cleanup that tries to remove all possible references
    """

    try:
        # Clear IPython output history first
        shell = get_ipython()
        shell.reset_selective()

        # Clear In/Out dictionaries
        shell.user_ns["In"] = [""]
        shell.user_ns["Out"] = {}

        # Clear any displayed outputs
        shell.display_pub.clear_output()

    except:
        print("Not running in IPython or error clearing namespace")

    # Get all objects in memory
    gc.collect()
    objects = gc.get_objects()

    # Count tensors we're trying to remove
    cuda_tensor_count = 0

    # First pass: try to delete tensors
    for obj in objects:
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensor_count += 1
                del obj
        except:
            continue

    # Clear any exception information
    sys.exc_clear() if hasattr(sys, "exc_clear") else None  # Python 2 compatibility

    # For IPython specifically
    try:
        shell = get_ipython()
        shell.user_ns["sys.last_traceback"] = None
        shell.user_ns["sys.last_type"] = None
        shell.user_ns["sys.last_value"] = None
    except:
        pass

    # Force garbage collection
    gc.collect()

    # Force garbage collection again
    gc.collect()

    # Calculate memory change
    final_memory = torch.cuda.memory_allocated()
    memory_freed = initial_memory - final_memory

    print(f"\nAttempted to clean {cuda_tensor_count} CUDA tensors")
    print(f"Initial memory: {initial_memory / 1024**2:.2f} MB")
    print(f"Final memory: {final_memory / 1024**2:.2f} MB")
    print(f"Memory freed: {memory_freed / 1024**2:.2f} MB")
    print(f"Memory still cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    return cuda_tensor_count, memory_freed

super_aggressive_cuda_cleanup()

# %%

results = []
cell_counts = [800]
n_runs = 1

print("GPU memory usage:", GPUtil.getGPUs()[0].memoryUsed)
gc.collect()



# %%

for cell_count in cell_counts:
    print(f"Benchmarking with {cell_count} cells...")
    # Subsample the data
    data_sub = X[np.random.choice(X.shape[0], cell_count, replace=False)]
    
    # Remove columns that no longer have any values
    idx_valid = data_sub.X.sum(axis=0) != 0
    data_sub = data_sub[:, idx_valid]

    for algorithm in ['pf2', 'Harmony']:
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs} for {algorithm}")

            data_run = data_sub.copy()  # Copy data to ensure consistent state
            # Run the benchmark
            metrics = benchmark_algorithm(data_run, algorithm, random_state=run)

            results.append({
                'cell_count': cell_count,
                'algorithm': algorithm,
                'run': run,
                'runtime': metrics['runtime'],
                'max_cpu_memory': metrics['max_cpu_memory'],
                'max_gpu_memory': metrics['max_gpu_memory']
            })

# Save the results to a CSV file
df_results = pd.DataFrame(results)
output_path = DATA_DIR / "benchmark_results.csv"
df_results.to_csv(output_path, index=False)
print(f"Benchmarking complete. Results saved to '{output_path}'.")
