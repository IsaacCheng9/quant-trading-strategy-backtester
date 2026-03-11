"""
A script to visualise the results of manual benchmarking between the pandas
implementation and the Polars implementation of the quant trading strategy
backtester.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_and_process_benchmark_data(
    csv_path: str,
) -> tuple[list, np.floating, np.floating]:
    """
    Loads benchmark data from a CSV file and processes it.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A tuple containing the sorted list of runtimes, mean, and standard
        deviation.
    """
    df = pl.read_csv(csv_path)
    times = df["runtime_seconds"].to_list()
    # Sort times in descending order so we can compare best and worst runs.
    times.sort(reverse=True)
    return times, np.mean(times), np.std(times)


def visualise_benchmark_times(
    benchmark_platform: str, pandas_csv: str, polars_csv: str
) -> None:
    """
    Creates an interactive bar chart comparing pandas and Polars execution
    times using Plotly.

    Args:
        benchmark_platform: The platform on which the benchmark was run
        pandas_csv: Path to CSV file with pandas execution times
        polars_csv: Path to CSV file with Polars execution times
    """
    # Read and process execution times from CSV files
    pandas_times, pandas_mean, pandas_std = load_and_process_benchmark_data(pandas_csv)
    polars_times, polars_mean, polars_std = load_and_process_benchmark_data(polars_csv)

    # Ensure that the number of runs is equal for fairness
    assert len(pandas_times) == len(polars_times), (
        "Different number of runs for pandas and Polars benchmarks"
    )
    num_runs = len(pandas_times)
    run_num_labels = [f"Run {i + 1}" for i in range(num_runs)]

    # Calculate average speed-up.
    avg_speedup = (pandas_mean / polars_mean - 1) * 100
    fig = go.Figure(
        data=[
            go.Bar(
                name="pandas",
                x=run_num_labels,
                y=pandas_times,
                marker_color="#FF4136",
                text=[f"{t:.2f}s" for t in pandas_times],
                textposition="outside",
                textfont=dict(size=18),
            ),
            go.Bar(
                name="Polars",
                x=run_num_labels,
                y=polars_times,
                marker_color="#0074D9",
                text=[f"{t:.2f}s" for t in polars_times],
                textposition="outside",
                textfont=dict(size=18),
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=dict(
            text=(
                f"pandas vs. Polars: Pairs Trading with"
                f" Ticker-Pair and Parameter Optimisation"
                f" ({benchmark_platform})<br>"
                f"<sub>Average Speed-Up:"
                f" {avg_speedup:.1f}% | "
                f"pandas: {pandas_mean:.2f}s"
                f" (std {pandas_std:.2f}s) | "
                f"Polars: {polars_mean:.2f}s"
                f" (std {polars_std:.2f}s)</sub>"
            ),
            font=dict(size=28),
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        xaxis_title=dict(text="Run Number", font=dict(size=20)),
        yaxis_title=dict(text="Execution Time (seconds)", font=dict(size=20)),
        # Leave room above the tallest bar for text labels.
        yaxis=dict(range=[0, max(pandas_times) * 1.15]),
        margin=dict(t=130),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1,
            font=dict(size=16),
        ),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=16))
    fig.update_yaxes(gridcolor="#eee", tickfont=dict(size=16))

    # Save the plot as an interactive HTML file.
    resources_dir = PROJECT_ROOT / "resources"
    platform_prefix = benchmark_platform.lower().replace(" ", "_")
    fig.write_html(str(resources_dir / f"{platform_prefix}_benchmark_results.html"))
    fig.show()


if __name__ == "__main__":
    PANDAS_AND_POLARS_BENCHMARKS = {
        "M1 Max": [
            str(PROJECT_ROOT / "resources/pandas_pairs_trading_benchmark.csv"),
            str(PROJECT_ROOT / "resources/polars_pairs_trading_benchmark.csv"),
        ],
    }
    for platform, (
        pandas_csv_file,
        polars_csv_file,
    ) in PANDAS_AND_POLARS_BENCHMARKS.items():
        visualise_benchmark_times(platform, pandas_csv_file, polars_csv_file)
