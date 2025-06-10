import os
import numpy as np
import time
import random
import networkx as nx
import glob
import pandas as pd
import matplotlib.pyplot as plt
from networkx.readwrite import read_graphml
from graph_info_printer_and_utils import GraphUtils
from arborescences import GreedyArborescenceDecomposition
from priority_list import PriorityList
from super_trees import SuperTreesArborescenceDecomposition
from beacon_nodes import beacon_nodes_algorithm
from global_controller import global_controller_generator

# ========== Already Provided Utilities ==========
def make_bidirectional_trees(T):
    return [tree.to_undirected() for tree in T]

def is_edge_failed(u, v, fails):
    return (u, v) in fails or (v, u) in fails

def clean_tree_of_fails(tree, fails):
    G = tree.copy()
    for u, v in list(G.edges()):
        if is_edge_failed(u, v, fails):
            G.remove_edge(u, v)
    return G



def route_with_switches(s, d, T, fails):
    undirected_T = [tree.to_undirected() for tree in T]
    cleaned_T = [clean_tree_of_fails(tree, fails) for tree in undirected_T]
    visited = set()

    queue = [(s, [], 0, 0, -1)]  # (current_node, path, hops, switches, current_tree)

    while queue:
        node, path, hops, switches, cur_tree = queue.pop(0)
        path = path + [node]
        visited.add((node, cur_tree))

        if node == d:
            return {
                "success": True,
                "path": path,
                "hops": hops,
                "switches": switches,
                "detour_edges": [(path[i], path[i+1]) for i in range(len(path) - 1)],
                "final_tree": cur_tree
            }

        tree_range = [cur_tree] if cur_tree != -1 else range(len(cleaned_T))

        for t_idx in tree_range:
            tree = cleaned_T[t_idx]
            if node not in tree:
                continue

            # --- PRIORITY: Follow PathToRoot if available and valid ---
            path_to_root = tree.nodes[node].get('PathToRoot', None)
            if path_to_root is not None:
                #print(f"PATHING TO ROOT ENTERED on node {node} leading to node {tree.nodes[node].get('PathToRoot')}")
                if path_to_root in tree and not is_edge_failed(node, path_to_root, fails):
                    if (path_to_root, t_idx) not in visited:
                        queue.insert(0, (path_to_root, path, hops + 1, switches, t_idx))  # Prioritize by inserting at front
                    continue  # Skip normal neighbors; we’re committing to PathToRoot

            # --- Normal neighbors ---
            for neighbor in tree.neighbors(node):
                if is_edge_failed(node, neighbor, fails):
                    continue
                if (neighbor, t_idx) not in visited:
                    queue.append((neighbor, path, hops + 1, switches, t_idx))

        # --- Tree switching ---
        if cur_tree != -1:
            for new_idx in range(len(cleaned_T)):
                if new_idx != cur_tree and node in cleaned_T[new_idx]:
                    if (node, new_idx) not in visited:
                        queue.append((node, path, hops, switches + 1, new_idx))

    return {
        "success": False,
        "reason": "No path found",
        "path": [],
        "hops": 0,
        "switches": 0,
        "detour_edges": []
    }


# ========== Evaluation Framework ==========
def evaluate_algorithm(graph, root, algorithm_fn, routing_fn, failure_prob=0.1, sample_size=100, measure_depth=False):
    results = {
        "runtime": None,  # Tree construction time
        "total_run_runtime": None,  # Full eval time including routing
        "stretch": [],
        "effectiveness": {"success": 0, "fail": 0},
        "functionality": True,
        "depths": [],
        "errors": [],
    }

    try:
        total_start_time = time.perf_counter()

        # --- Tree Construction Timing ---
        start_time = time.perf_counter()
        trees = algorithm_fn(graph)
        results["runtime"] = (time.perf_counter() - start_time) * 1000  # ms

        if not trees or not isinstance(trees, list):
            raise ValueError("Algorithm did not return a valid list of trees.")

        if  measure_depth:
            for tree in trees:
                if root not in tree.nodes:
                    results["errors"].append("Root not in tree")
                    continue
                if not nx.is_connected(tree.to_undirected()):
                    results["errors"].append("Tree is not connected")
                    continue
                try:
                    depths = nx.single_source_shortest_path_length(tree.to_undirected(), root)
                    max_depth = max(depths.values())
                    results["depths"].append(max_depth)
                   # print(f"Tree with {len(tree.nodes)} nodes — Max Depth: {max_depth}")
                except Exception as e:
                    results["errors"].append(f"Depth error: {str(e)}")

        nodes = list(graph.nodes())
        sources = [n for n in nodes if n != root]

        for _ in range(sample_size):
            s = random.choice(sources)
            d = root  # always destination is root
            fails = [edge for edge in graph.edges() if random.random() < failure_prob]

            try:
                outcome = routing_fn(s, d, trees, fails)

                if outcome["success"]:
                    results["effectiveness"]["success"] += 1
                    try:
                        orig_dist = nx.shortest_path_length(graph, s, d)
                        if orig_dist > 0:
                            if outcome["hops"] >= orig_dist:
                                stretch = (outcome["hops"] - orig_dist) #/ orig_dist
                                results["stretch"].append(stretch)
                                print(stretch)
                    except nx.NetworkXNoPath:
                        pass
                else:
                    results["effectiveness"]["fail"] += 1
            except Exception as e:
                results["effectiveness"]["fail"] += 1
                results["errors"].append(f"Routing error: {str(e)}")

        results["total_run_runtime"] = (time.perf_counter() - total_start_time) * 1000  # ms

    except Exception as e:
        results["functionality"] = False
        error_msg = f"Fatal error: {str(e)}"
        print(error_msg)
        results["errors"].append(error_msg)

    return results



def run_all_algorithms_on_benchmarks(algorithms: dict, graph_files, failure_prob=0.1, sample_size=100, measure_depth=False):
    all_results = {}

    for file in graph_files:
        graph = read_graphml(file)
        graph = graph.to_undirected()
        graph = nx.convert_node_labels_to_integers(graph)
        GraphUtils.assign_random_survivability(graph)
        nod = list(graph.nodes())
        graph.graph['root'] = random.sample(nod, 1)[0]
        graph.graph['k'] = 3
        root = graph.graph['root']
        all_results[file] = {}

        for name, algorithm_fn in algorithms.items():
            print(f"Evaluating {name} on {file}...")
            res = evaluate_algorithm(
                graph, root,
                algorithm_fn=algorithm_fn,
                routing_fn=route_with_switches,
                failure_prob=failure_prob,
                sample_size=sample_size,
                measure_depth=measure_depth
            )            

            all_results[file][name] = res
    return all_results
def summarize_multiple_runs(all_run_results, output_dir="plots", save_plots=True):
    combined_data = []

    for run_index, all_results in enumerate(all_run_results):
        for graph_file, alg_results in all_results.items():
            # Load the graph to count the nodes
            graph = nx.read_graphml(graph_file)
            graph = graph.to_undirected()
            graph = nx.convert_node_labels_to_integers(graph)
            num_nodes = graph.number_of_nodes()

            for alg, res in alg_results.items():
                stretch_vals = res["stretch"]
                row = {
                    "run": run_index,
                    "graph": os.path.basename(graph_file),
                    "num_nodes": num_nodes,
                    "algorithm": alg,
                    "runtime": round(res["runtime"], 4),
                    "total_run_runtime": round(res["total_run_runtime"], 4),
                    "median_stretch": round(np.median(stretch_vals), 4) if stretch_vals else None,
                    "success_rate": res["effectiveness"]["success"] / (res["effectiveness"]["success"] + res["effectiveness"]["fail"]),
                    "avg_depth": round(sum(res["depths"]) / len(res["depths"]), 2) if res["depths"] else None,
                    "errors": len(res["errors"]),
                }
                combined_data.append(row)

    df = pd.DataFrame(combined_data)

    # --- Save raw combined data ---
    os.makedirs(output_dir, exist_ok=True)
    combined_csv_path = os.path.join(output_dir, "raw_combined_results.csv")
    df.to_csv(combined_csv_path, index=False)
    print(f"Raw results CSV saved to: {combined_csv_path}")

    # --- Aggregate metrics (average and std) across runs ---
    #metric_cols = ["runtime", "total_run_runtime", "success_rate", "median_stretch", "avg_depth"]
    # Compute means for the other metrics
    metric_cols = ["runtime", "total_run_runtime", "success_rate", "avg_depth"]
    avg_df = df.groupby(["graph", "num_nodes", "algorithm"])[metric_cols].mean().reset_index()
    avg_df = avg_df.round(4)

    median_stretch_df = df.groupby(["graph", "num_nodes", "algorithm"])["median_stretch"].max().reset_index()
    median_stretch_df = median_stretch_df.round(4)

    # Use outer join to make sure no data gets dropped if there's a mismatch
    final_df = pd.merge(avg_df, median_stretch_df, on=["graph", "num_nodes", "algorithm"], how="outer")

    # Save to CSV
    final_csv_path = os.path.join(output_dir, "final_metrics.csv")
    final_df.to_csv(final_csv_path, index=False)
    print(f"Final metrics CSV saved to: {final_csv_path}")

    # --- Plotting scatter plots ---
    if save_plots:
     # --- Plotting scatter plots ---
        def scatter_plot(metric, ylabel, title, filename):
            plt.figure(figsize=(12, 8))
            algorithms = final_df["algorithm"].unique()
            colors = plt.cm.tab10.colors  # Up to 10 colors

            for i, alg in enumerate(algorithms):
                sub_df = final_df[final_df["algorithm"] == alg]
                plt.scatter(sub_df["num_nodes"], sub_df[metric], label=alg, color=colors[i % len(colors)], alpha=0.7, s=40)

            plt.xlabel("Number of Nodes")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), format="svg")  # 👈 Explicitly SVG
            plt.close()
    # --- Calls with filenames ending in .svg ---
    scatter_plot("runtime", "Runtime (ms)", "Runtime vs. Number of Nodes", "runtime_vs_nodes.svg")
    scatter_plot("total_run_runtime", "Total Runtime (ms)", "Total Runtime vs. Number of Nodes", "total_runtime_vs_nodes.svg")
    scatter_plot("median_stretch", "Median Stretch", "Median Stretch vs. Number of Nodes", "median_stretch_vs_nodes.svg")
    scatter_plot("success_rate", "Success Rate", "Success Rate vs. Number of Nodes", "success_rate_vs_nodes.svg")
    scatter_plot("avg_depth", "Average Depth", "Average Depth vs. Number of Nodes", "avg_depth_vs_nodes.svg")

    print("SVG scatter plots saved in:", output_dir)

# ========== Main Execution ==========
import multiprocessing as mp

def run_single_run(run_index, algorithms, graph_files, failure_prob, sample_size, measure_depth):
    print(f"=== Run {run_index + 1} ===")
    return run_all_algorithms_on_benchmarks(
        algorithms,
        graph_files,
        failure_prob=failure_prob,
        sample_size=sample_size,
        measure_depth=measure_depth
    )


if __name__ == "__main__":
    TOTAL_RUNS = 32  # Total runs you want
    NUM_PARALLEL_PROCESSES = 8  # Limit to available cores

    algorithms = {
        "PrioList": PriorityList.prio_list_output,
        "Super Trees": SuperTreesArborescenceDecomposition,
        "Beacon Nodes": beacon_nodes_algorithm,
        "Global Controller": global_controller_generator, 
        "Simple Decomposition": GreedyArborescenceDecomposition
    }

    benchmark_folder = "benchmark_graphs"
    graph_files = glob.glob(os.path.join(benchmark_folder, "*.graphml"))

    failure_prob = 0.1
    sample_size = 100
    measure_depth = True

    # Prepare all runs
    pool_args = [
        (i, algorithms, graph_files, failure_prob, sample_size, measure_depth)
        for i in range(TOTAL_RUNS)
    ]

    # Use a pool with NUM_PARALLEL_PROCESSES workers
    with mp.Pool(processes=NUM_PARALLEL_PROCESSES) as pool:
        all_run_results = pool.starmap(run_single_run, pool_args)

    # Summarize results as usual
    summarize_multiple_runs(all_run_results, output_dir="plots", save_plots=True)