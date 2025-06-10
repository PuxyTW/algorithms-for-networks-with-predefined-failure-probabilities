import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_from_csv(csv_path, metric_name, output_dir="plots", node_range=None, show=False):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if node_range:
        min_nodes, max_nodes = node_range
        df = df[(df["nodes"] >= min_nodes) & (df["nodes"] <= max_nodes)]

    algorithms = df["algorithm"].unique()
    node_sizes = sorted(df["nodes"].unique())

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    for alg in algorithms:
        sub_df = df[df["algorithm"] == alg].set_index("nodes").sort_index()
        mean_col = f"{metric_name}_mean"
        if mean_col not in sub_df.columns:
            print(f"[ERROR] Column '{mean_col}' not found in CSV.")
            return

        ax.plot(
            sub_df.index,
            sub_df[mean_col],
            label=alg,
            marker="o",
            linestyle="-"
        )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} vs Nodes")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    else:
        out_file = os.path.join(output_dir, f"{metric_name}_vs_nodes.svg")
        plt.savefig(out_file, format="svg")
        print(f"[Saved] {out_file}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a specific metric from its summary CSV.")
    parser.add_argument("--csv", required=True, help="Path to the metric summary CSV (e.g. runtime_summary.csv)")
    parser.add_argument("--metric", required=True, help="Metric name (e.g. runtime, avg_depth, etc.)")
    parser.add_argument("--output", default="plots", help="Output directory for plot SVGs")
    parser.add_argument("--min_nodes", type=int, help="Minimum node count to display")
    parser.add_argument("--max_nodes", type=int, help="Maximum node count to display")
    parser.add_argument("--show", action="store_true", help="Show plot interactively instead of saving")

    args = parser.parse_args()

    node_range = None
    if args.min_nodes is not None and args.max_nodes is not None:
        node_range = (args.min_nodes, args.max_nodes)

    plot_metric_from_csv(
        csv_path=args.csv,
        metric_name=args.metric,
        output_dir=args.output,
        node_range=node_range,
        show=args.show
    )
