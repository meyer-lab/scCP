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

from ..factorization import pf2
from ..imports import import_thomson
from .common import getSetup, subplotLabel

RECOMPUTE = False  # Set to True to rerun benchmarks, False to load from saved results

DATA_DIR = Path(__file__).parent.parent / "data"

def makeFigure():
    """Create figure comparing computational requirements of pf2 and Harmony."""
    ax, f = getSetup((15, 4), (1, 3))  # Adjusted to accommodate an additional subplot

    # Load or compute benchmark results
    if RECOMPUTE:
        cell_counts = [1000, 5000, 10000, 20000, 50000]
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
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes single GPU usage

    start_time = time.time()
    tracemalloc.start()

    if algorithm == 'pf2':
        # Run pf2 algorithm
        pf2(data, rank=rank, **kwargs)
    elif algorithm == 'Harmony':
        # Run Harmony algorithm using harmonypy
        from harmonypy import run_harmony

        # Ensure data.X is a dense array
        X = data.X.toarray() if hasattr(data.X, 'toarray') else data.X

        # Prepare metadata
        if 'batch' not in data.obs:
            raise ValueError("Data must have 'batch' column in .obs for Harmony integration.")
        meta_data = data.obs[['batch']]

        # Run Harmony with GPU acceleration (if supported)
        Z_corr = run_harmony(X, meta_data, vars_use=['batch'])

        # Update data.X with the corrected data
        data.X = Z_corr
    else:
        raise ValueError("Algorithm must be 'pf2' or 'Harmony'")

    # Measure CPU memory usage
    current, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    # Measure GPU memory usage
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    max_gpu_memory = mem_info.used / 1e6  # Convert bytes to megabytes

    pynvml.nvmlShutdown()

    runtime = end_time - start_time
    max_cpu_memory = peak_cpu / 1e6  # Convert bytes to megabytes

    return {
        'runtime': runtime,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory
    }


def run_benchmarks(cell_counts: List[int], n_runs: int = 3):
    """
    Runs benchmarks for different cell counts and saves the results.

    Parameters:
        cell_counts: List of cell counts to test.
        n_runs: Number of runs for averaging.
    """
    results = []

    # Load the full Thomson dataset
    X = import_thomson()

    for cell_count in cell_counts:
        print(f"Benchmarking with {cell_count} cells...")
        # Subsample the data
        data_sub = X[:cell_count].copy()

        for algorithm in ['pf2', 'Harmony']:
            runtimes = []
            max_cpu_memories = []
            max_gpu_memories = []

            for run in range(n_runs):
                print(f"Run {run + 1}/{n_runs} for {algorithm}")
                data_run = data_sub.copy()  # Copy data to ensure consistent state
                # Run the benchmark
                metrics = benchmark_algorithm(data_run, algorithm, random_state=run)
                runtimes.append(metrics['runtime'])
                max_cpu_memories.append(metrics['max_cpu_memory'])
                max_gpu_memories.append(metrics['max_gpu_memory'])

            avg_runtime = np.mean(runtimes)
            avg_cpu_memory = np.mean(max_cpu_memories)
            avg_gpu_memory = np.mean(max_gpu_memories)

            results.append({
                'cell_count': cell_count,
                'algorithm': algorithm,
                'avg_runtime': avg_runtime,
                'avg_cpu_memory': avg_cpu_memory,
                'avg_gpu_memory': avg_gpu_memory
            })

    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    output_path = DATA_DIR / "benchmark_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Benchmarking complete. Results saved to '{output_path}'.")


if __name__ == "__main__":
    # Define the cell counts to benchmark
    cell_counts = [1000, 5000, 10000, 20000, 50000]  # Adjust based on dataset size

    # Run the benchmarks
    run_benchmarks(cell_counts)