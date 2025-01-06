import os
import time
import tracemalloc

import anndata
import cupy as cp
import numpy as np
import pandas as pd
import scanorama
import scvi
import seaborn as sns
import torch
from harmonypy import run_harmony

from sccp.factorization import pf2
from sccp.figures.common import getSetup, subplotLabel
from sccp.imports import import_lupus

RECOMPUTE = False  # Set to True to run benchmarks, False to skip
UPDATE_EXISTING_RESULTS = (
    True  # If True, replace matching rows in the CSV; if False, rewrite the file
)
DATA_PATH = "/opt/pf2/benchmark_compute_requirement.csv"


def makeFigure():
    ax, f = getSetup((7, 3), (1, 2))  # Adjusted to accommodate an additional subplot

    # Load or compute benchmark results
    if RECOMPUTE:
        cell_counts = [int(10**x) for x in [4, 4.5, 5, 5.5, 6]]
        run_benchmarks(cell_counts)

    # Load results
    df_results = pd.read_csv(DATA_PATH)

    # Runtime comparison
    sns.lineplot(
        data=df_results,
        x="cell_count",
        y="runtime",
        hue="algorithm",
        ax=ax[0],
        marker="o",
    )
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Number of Cells")
    ax[0].set_ylabel("Runtime (seconds)")
    ax[0].set_title("Runtime Comparison")
    ax[0].legend(title="")

    # Compute total memory
    df_results["total_memory"] = (
        df_results["max_cpu_memory"] + df_results["max_gpu_memory"]
    )

    # Melt the DataFrame to long format for plotting
    df_melted = df_results.melt(
        id_vars=["cell_count", "algorithm", "run"],
        value_vars=["max_cpu_memory", "max_gpu_memory", "total_memory"],
        var_name="memory_type",
        value_name="memory_usage",
    )

    # plot total memory
    total_memory_data = df_melted[df_melted["memory_type"].isin(["total_memory"])]
    sns.lineplot(
        data=total_memory_data,
        x="cell_count",
        y="memory_usage",
        hue="algorithm",
        style="memory_type",
        ax=ax[1],
        marker="o",
    )
    ax[1].legend([])
    ax[1].set_xlabel("Number of Cells")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Total Memory Usage (bytes)")
    ax[1].set_title("Total Memory Usage Comparison")

    subplotLabel(ax)

    return f


def benchmark_algorithm(
    data: anndata.AnnData, algorithm: str, rank: int = 20, **kwargs
) -> dict[str, float]:
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

    if algorithm == "pf2":
        # Run pf2 algorithm
        assert data.shape[0] > rank, "Number of cells must be greater than rank"
        data.obs["condition_unique_idxs"] = data.obs["condition_unique_idxs"].astype(
            "category"
        )

        start_time = time.time()
        tracemalloc.start()
        pf2(data, rank=rank, doEmbedding=False, **kwargs)

        # Measure GPU memory usage
        pool = cp.get_default_memory_pool()
        max_gpu_memory = pool.total_bytes()
        pool.free_all_blocks()

    elif algorithm == "Harmony":
        # Ensure data.X is a dense array
        X = data.X.toarray()

        # Prepare metadata
        if "pool" not in data.obs:
            raise ValueError(
                "Data must have 'batch' column in .obs for Harmony integration."
            )
        meta_data = data.obs[["pool"]]

        start_time = time.time()
        tracemalloc.start()

        run_harmony(X, meta_data, vars_use=["pool"])

        max_gpu_memory = 0  # GPU not supported

    elif algorithm == "scVI":
        # Preprocess data for scVI
        data.layers["counts"] = data.X.copy()
        del data.X

        # Setup scVI anndata
        start_time = time.time()
        tracemalloc.start()
        scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key="pool")

        # Track total GPU memory usage
        model = scvi.model.SCVI(data)
        model.train()
        max_gpu_memory = torch.cuda.max_memory_reserved()

    elif algorithm == "Scanorama":
        # Ensure data.X is a dense array
        X_list = [
            data[data.obs["pool"] == batch].X.toarray()
            if hasattr(data.X, "toarray")
            else data[data.obs["pool"] == batch].X
            for batch in data.obs["pool"].unique()
        ]
        genes_list = [data.var_names] * len(X_list)
        del data

        start_time = time.time()
        tracemalloc.start()
        # Run Scanorama integration
        integrated, genes = scanorama.integrate(X_list, genes_list)

        max_gpu_memory = 0  # GPU not supported
    else:
        raise ValueError("Invalid algorithm")

    # Measure CPU memory usage
    current, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    runtime = end_time - start_time
    max_cpu_memory = peak_cpu

    return {
        "runtime": runtime,
        "max_cpu_memory": max_cpu_memory,
        "max_gpu_memory": max_gpu_memory,
    }


def run_benchmarks(cell_counts: list[int], n_runs: int = 1):
    """
    Runs benchmarks for different cell counts and saves the results after each
    algorithm. If RECOMPUTE and UPDATE_EXISTING_RESULTS are both True, previously
    saved results are loaded and only rows matching the current (algorithm, cell_count)
    are updated. Otherwise, the entire CSV file is rewritten (if RECOMPUTE=True).
    """

    X = import_lupus(geneThreshold=0.005)
    print(f"{X.X.shape=}")

    # Decide whether to do selective updating or overwrite everything
    if RECOMPUTE and UPDATE_EXISTING_RESULTS and os.path.exists(DATA_PATH):
        # Load existing results to selectively update them
        existing_df = pd.read_csv(DATA_PATH)
        all_results = existing_df.to_dict("records")
    else:
        # Either RECOMPUTE=False or we want to overwrite, or file doesn't exist
        all_results = []

    for cell_count in cell_counts:
        print(f"Benchmarking with {cell_count} cells...")
        # Subsample the data
        data_sub = X[np.random.choice(X.shape[0], cell_count, replace=False)]

        # Remove columns that no longer have any values
        idx_valid = data_sub.X.sum(axis=0) != 0
        data_sub = data_sub[:, idx_valid]
        print(f"{data_sub.X.shape=}")

        for algorithm in ["pf2"]:
            results = []
            for run in range(n_runs):
                print(f"Run {run + 1}/{n_runs} for {algorithm}")

                data_run = data_sub.copy()  # Copy data to ensure consistent state
                metrics = benchmark_algorithm(data_run, algorithm, random_state=run)

                results.append(
                    {
                        "cell_count": cell_count,
                        "algorithm": algorithm,
                        "run": run,
                        "runtime": metrics["runtime"],
                        "max_cpu_memory": metrics["max_cpu_memory"],
                        "max_gpu_memory": metrics["max_gpu_memory"],
                    }
                )

            if RECOMPUTE and UPDATE_EXISTING_RESULTS:
                # Remove previous entries matching this algorithm & cell_count
                all_results = [
                    row
                    for row in all_results
                    if not (
                        row["algorithm"] == algorithm
                        and row["cell_count"] == cell_count
                    )
                ]

            # Append current results regardless of RECOMPUTE or not
            # (If RECOMPUTE=False, we wouldn't be here at all)
            all_results.extend(results)

            # Save to CSV after each algorithm
            pd.DataFrame(all_results).to_csv(DATA_PATH, index=False)
            print(f"Results after algorithm '{algorithm}' saved to '{DATA_PATH}'.")

    print("Benchmarking complete.")
