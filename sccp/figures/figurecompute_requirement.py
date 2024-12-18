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
import scvi
import cupy as cp
import scanorama
import torch

from harmonypy import run_harmony

from sccp.factorization import pf2
from sccp.imports import import_lupus
from sccp.figures.common import getSetup, subplotLabel


RECOMPUTE = False  # Set to True to rerun benchmarks, False to load from saved results

DATA_DIR = Path(__file__).parent.parent / "data"

def makeFigure():
    ax, f = getSetup((7, 3), (1, 2))  # Adjusted to accommodate an additional subplot

    # Load or compute benchmark results
    if RECOMPUTE:
        cell_counts = [int(x) for x in [1e3, 1e4, 1e5, 1e6, 1e7]]
        run_benchmarks(cell_counts)

    # Load results
    results_path = DATA_DIR / "benchmark_results.csv"
    df_results = pd.read_csv(results_path)

    # Runtime comparison
    sns.lineplot(
        data=df_results,
        x="cell_count",
        y="runtime",
        hue="algorithm",
        ax=ax[0],
        marker='o'
    )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Number of Cells")
    ax[0].set_ylabel("Runtime (seconds)")
    ax[0].set_title("Runtime Comparison")
    ax[0].legend(title="Algorithm")

    # Compute total memory
    df_results['total_memory'] = df_results['max_cpu_memory'] + df_results['max_gpu_memory']

    # Melt the DataFrame to long format for plotting
    df_melted = df_results.melt(
        id_vars=['cell_count', 'algorithm', 'run'],
        value_vars=['max_cpu_memory', 'max_gpu_memory', 'total_memory'],
        var_name='memory_type',
        value_name='memory_usage'
    )

    # Map the memory_type to more readable labels
    memory_type_mapping = {
        'max_cpu_memory': 'CPU Memory',
        'max_gpu_memory': 'GPU Memory',
        'total_memory': 'Total Memory'
    }
    df_melted['memory_type'] = df_melted['memory_type'].map(memory_type_mapping)

    # Define custom dash styles for memory types
    dashes_styles = {
        'Total Memory': '',       # Solid line
        'CPU Memory': (5, 5),     # Dashed line
        'GPU Memory': (2, 2)      # Dotted line
    }

    # First, create plot for total memory only
    total_memory_data = df_melted[df_melted['memory_type'] == 'Total Memory']
    sns.lineplot(
        data=total_memory_data,
        x='cell_count',
        y='memory_usage',
        hue='algorithm',
        ax=ax[1],
        marker='o'
    )
    ax[1].legend([])
    ax[1].set_xlabel("Number of Cells")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Total Memory Usage (bytes)")
    ax[1].set_title("Total Memory Usage Comparison")

    # # Then, create plot for CPU and GPU memory
    # cpu_gpu_data = df_melted[df_melted['memory_type'].isin(['CPU Memory', 'GPU Memory'])]
    # sns.lineplot(
    #     data=cpu_gpu_data,
    #     x='cell_count',
    #     y='memory_usage',
    #     hue='algorithm',
    #     style='memory_type',
    #     ax=ax[2],
    #     markers=True,
    #     dashes=[(5, 5), (2, 2)]  # Different dash patterns for CPU and GPU
    # )
    # ax[2].set_xscale("log")
    # ax[2].set_yscale("log")
    # ax[2].set_xlabel("Number of Cells")
    # ax[2].set_ylabel("Memory Usage (bytes)")
    # ax[2].set_title("CPU and GPU Memory Usage")

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


    if algorithm == 'pf2':
        # Run pf2 algorithm
        assert data.shape[0] > rank, "Number of cells must be greater than rank"

        start_time = time.time()
        tracemalloc.start()
        pf2(data, rank=rank, doEmbedding=False, **kwargs)

        # Measure GPU memory usage
        pool = cp.get_default_memory_pool()
        max_gpu_memory = pool.total_bytes()
        pool.free_all_blocks()

    elif algorithm == 'Harmony':
        # Ensure data.X is a dense array
        X = data.X.toarray()

        # Prepare metadata
        if 'pool' not in data.obs:
            raise ValueError("Data must have 'batch' column in .obs for Harmony integration.")
        meta_data = data.obs[['pool']]

        start_time = time.time()
        tracemalloc.start()

        run_harmony(X, meta_data, vars_use=['pool'])

        max_gpu_memory = 0 # GPU not supported
    
    elif algorithm == 'scVI':
        # Preprocess data for scVI
        data.layers["counts"] = data.X.copy()
        del data.X

        # Setup scVI anndata
        start_time = time.time()
        tracemalloc.start()
        scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key="pool")

        # Initialize and train the model
        model = scvi.model.SCVI(data)
        model.train()

        max_gpu_memory = torch.cuda.memory_allocated()
    
    elif algorithm == 'Scanorama':
        # Ensure data.X is a dense array
        X_list = [data[data.obs['pool'] == batch].X.toarray() if hasattr(data.X, 'toarray') else data[data.obs['pool'] == batch].X for batch in data.obs['pool'].unique()]
        genes_list = [data.var_names] * len(X_list)
        del data

        start_time = time.time()
        tracemalloc.start()
        # Run Scanorama integration
        integrated, genes = scanorama.integrate(X_list, genes_list)

        max_gpu_memory = 0 # GPU not supported
    else:
        raise ValueError("Invalid algorithm")

    # Measure CPU memory usage
    current, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    runtime = end_time - start_time
    max_cpu_memory = peak_cpu

    return {
        'runtime': runtime,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory
    }

def run_benchmarks(cell_counts: List[int], n_runs: int = 1):
    """
    Runs benchmarks for different cell counts and saves the results after each algorithm.

    Parameters:
        cell_counts: List of cell counts to test.
        n_runs: Number of runs for averaging.
    """

    X = import_lupus()

    # Initialize an empty list to collect all results
    all_results = []

    for cell_count in cell_counts:
        print(f"Benchmarking with {cell_count} cells...")
        # Subsample the data
        data_sub = X[np.random.choice(X.shape[0], cell_count, replace=False)]
        
        # Remove columns that no longer have any values
        idx_valid = data_sub.X.sum(axis=0) != 0
        data_sub = data_sub[:, idx_valid]

        for algorithm in ["pf2", "Harmony", "scVI", "Scanorama"]:
            results = []  # List to collect results for the current algorithm
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

            # Append the current algorithm's results to the overall results
            all_results.extend(results)

            # Save the results to a CSV file after each algorithm
            df_results = pd.DataFrame(all_results)
            output_path = DATA_DIR / "benchmark_results.csv"
            df_results.to_csv(output_path, index=False)
            print(f"Results after algorithm '{algorithm}' saved to '{output_path}'.")

    print("Benchmarking complete.")
