"""
A script to visualise the results of manual benchmarking between the pandas
implementation and the Polars implementation of the quant trading strategy
backtester.
"""

import plotly.graph_objects as go
import numpy as np
import polars as pl


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
    assert len(pandas_times) == len(
        polars_times
    ), "Different number of runs for pandas and Polars benchmarks"
    num_runs = len(pandas_times)
    run_num_labels = [f"Run {i+1}" for i in range(num_runs)]

    # Calculate average speed-up
    avg_speedup = (pandas_mean / polars_mean - 1) * 100
    fig = go.Figure(
        data=[
            go.Bar(
                name="pandas",
                x=run_num_labels,
                y=pandas_times,
                marker_color="red",
                opacity=0.80,
                error_y=dict(type="data", array=[pandas_std] * num_runs, visible=True),
            ),
            go.Bar(
                name="Polars",
                x=run_num_labels,
                y=polars_times,
                marker_color="blue",
                opacity=0.80,
                error_y=dict(type="data", array=[polars_std] * num_runs, visible=True),
            ),
        ]
    )
    # Change the bar mode
    fig.update_layout(
        barmode="group",
        title=(
            "pandas vs Polars: Pairs Trading with Ticker-Pair and Parameter "
            "Optimisation – Execution Time"
        ),
        xaxis_title="Run Number",
        yaxis_title="Execution Time (seconds)",
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"Average Speedup: {avg_speedup:.3f}%",
                showarrow=False,
                font=dict(size=14),
            )
        ],
    )
    # Add value labels on top of each bar
    for i in range(num_runs):
        fig.add_annotation(
            x=f"Run {i+1}",
            y=pandas_times[i],
            text=f"{pandas_times[i]:.4f}",
            showarrow=False,
            yshift=10,
        )
        fig.add_annotation(
            x=f"Run {i+1}",
            y=polars_times[i],
            text=f"{polars_times[i]:.4f}",
            showarrow=False,
            yshift=10,
        )
    # Add summary statistics
    fig.add_annotation(
        x=1,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"pandas: Mean = {pandas_mean:.4f}s, Std Dev = {pandas_std:.4f}s<br>"
        f"Polars: Mean = {polars_mean:.4f}s, Std Dev = {polars_std:.4f}s",
        showarrow=False,
        font=dict(size=12),
        align="right",
        xanchor="right",
        yanchor="top",
    )

    # Save the plot as an interactive HTML file
    fig.write_html(f"resources/{benchmark_platform}_benchmark_results.html")
    # Show the plot (if running in an environment that supports it)
    fig.show()


if __name__ == "__main__":
    PANDAS_AND_POLARS_BENCHMARKS = {
        "m1_max": [
            "resources/m1_max_pandas_pairs_trading_with_optimisers.csv",
            "resources/m1_max_polars_pairs_trading_with_optimisers.csv",
        ],
        "streamlit": [
            "resources/streamlit_pandas_pairs_trading_with_optimisers.csv",
            "resources/streamlit_polars_pairs_trading_with_optimisers.csv",
        ],
    }
    for platform, (
        pandas_csv_file,
        polars_csv_file,
    ) in PANDAS_AND_POLARS_BENCHMARKS.items():
        visualise_benchmark_times(platform, pandas_csv_file, polars_csv_file)
