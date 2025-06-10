import os
import numpy as np
import time
import random
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from graph_info_printer_and_utils import GraphUtils
from arborescences import GreedyArborescenceDecomposition
from priority_list import PriorityList
from super_trees import SuperTreesArborescenceDecomposition
from beacon_nodes import beacon_nodes_algorithm
from global_controller import global_controller_generator
from multiprocessing import Pool, cpu_count
import os

params = {
    "failure_around_root": 0.005,
    "failure_on_leaves": 0.1,
    "failure_in_range_min": 0.05,
    "failure_in_range_max": 0.1,     
    }

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

import networkx as nx




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
            fails = [
                    (u, v) for u, v in graph.edges()
                    if random.random() < graph.edges[u, v].get('failure', 0.0)
                ]            
            #print(f"There are {len(fails)} fails in this graph")
            try:
                outcome = routing_fn(s, d, trees, fails)

                if outcome["success"]:
                    results["effectiveness"]["success"] += 1
                    try:
                        orig_dist = nx.shortest_path_length(graph, s, d)                       
                        if orig_dist > 0:
                            if outcome["hops"] < orig_dist:
                                print(f"[WARNING] hops={outcome['hops']} < orig_dist={orig_dist} for ({s}, {d})")
                            if outcome["hops"] >= orig_dist:
                                stretch = (outcome["hops"] - orig_dist) #/ orig_dist
                                results["stretch"].append(stretch)
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



# ========== Graph Generation ==========

def generate_random_graphs(node_sizes, degree=3, graphs_per_size=3):
    generated_graphs = []

    for n in node_sizes:
        for i in range(graphs_per_size):
            if degree >= n:
                continue  # Degree must be less than node count
            try:
                G = nx.random_regular_graph(degree, n)
                G = nx.convert_node_labels_to_integers(G)
                G.graph['name'] = f"rrg_n{n}_g{i}"

                # ✅ Check connectivity
                if not nx.is_connected(G):
                    print(f"⚠️ Warning: Graph {G.graph['name']} is NOT connected.")

                generated_graphs.append(G)
            except Exception as e:
                print(f"❌ Error generating graph with n={n}, d={degree}: {e}")
    return generated_graphs
#                                                                ========== Algorithm Evaluation ==========

def run_all_algorithms_on_synthetic_graphs(algorithms: dict, synthetic_graphs, failure_prob=0.1, sample_size=100, measure_depth=True):
    all_results = {}

    for graph in synthetic_graphs:
        graph = graph.to_undirected()
        #GraphUtils.assign_random_survivability(graph)
        nod = list(graph.nodes())
        graph.graph['root'] = random.sample(nod, 1)[0]
        graph.graph['k'] = 3
        root = graph.graph['root']
        graph_id = graph.graph['name']
        GraphUtils.assign_specific_failure_probabilities(graph, params)
        all_results[graph_id] = {}

        for name, algorithm_fn in algorithms.items():
            print(f"Evaluating {name} on {graph_id}...")
            res = evaluate_algorithm(
                graph, root,
                algorithm_fn=algorithm_fn,
                routing_fn=route_with_switches,
                failure_prob=failure_prob,
                sample_size=sample_size,
                measure_depth=measure_depth
            )
            all_results[graph_id][name] = res
    return all_results

#                                                           ========== Metrics Aggregation and Plotting ==========
def summarize_multiple_runs(all_run_results, output_dir="plots", save_plots=True):
    combined_data = []

    for run_index, all_results in enumerate(all_run_results):
        for graph_file, alg_results in all_results.items():
            node_match = int(graph_file.split("_")[1][1:])  # 'n100_g0' → 100
            for alg, res in alg_results.items():
                stretch_vals = res["stretch"]
                row = {
                    "run": run_index,
                    "graph": graph_file,
                    "nodes": node_match,
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

    metric_cols = ["runtime", "total_run_runtime", "success_rate", "median_stretch", "avg_depth"]

    agg_funcs = {
        "runtime": "mean",
        "total_run_runtime": "mean",
        "success_rate": "mean",
        "avg_depth": "mean",
        "median_stretch": "max",    # 👈 median_stretch will use max
    }

    avg_df = df.groupby(["nodes", "algorithm"]).agg(agg_funcs).reset_index()
    std_df = df.groupby(["nodes", "algorithm"])[metric_cols].std().reset_index()

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "raw_combined_results.csv"), index=False)
    avg_df.to_csv(os.path.join(output_dir, "average_metrics.csv"), index=False)
    std_df.to_csv(os.path.join(output_dir, "std_metrics.csv"), index=False)

    # Plotting
    if save_plots:

        def plot_metric_over_nodes(metric_name, ylabel, title, filename, agg_func="mean"):
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            node_sizes = sorted(df["nodes"].unique())
            algorithms = sorted(df["algorithm"].unique())

            for alg in algorithms:
                vals = []
                for n in node_sizes:
                    group = df[(df["nodes"] == n) & (df["algorithm"] == alg)][metric_name]
                    if not group.empty:
                        if metric_name == "median_stretch":
                            val = group.max()  # 👈 Use max for plotting stretch
                        else:
                            val = getattr(group, agg_func)()
                        vals.append(val)
                    else:
                        vals.append(None)

                ax.plot(node_sizes, vals, marker='o', label=alg)

            ax.set_xlabel("Number of Nodes")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), format="svg")
            plt.close()

        plot_metric_over_nodes('runtime', 'Runtime (ms)', 'Algorithm Runtime vs Nodes', 'runtime_vs_nodes.svg')
        plot_metric_over_nodes('success_rate', 'Success Rate', 'Success Rate vs Nodes', 'success_rate_vs_nodes.svg')
        plot_metric_over_nodes('median_stretch', 'Max Median Stretch', 'Max Median Stretch vs Nodes', 'stretch_vs_nodes.svg')
        plot_metric_over_nodes('avg_depth', 'Average Depth', 'Average Depth vs Nodes', 'depth_vs_nodes.svg')
        plot_metric_over_nodes('total_run_runtime', 'Total Runtime (ms)', 'Total Runtime vs Nodes', 'total_runtime_vs_nodes.svg')

# ========== Main Execution ==========

# ✅ Define this at top-level
def one_run(run_idx):
    print(f"=== Synthetic Run {run_idx+1} (PID {os.getpid()}) ===")
    
    NODE_SIZES = [10, 24, 50, 80, 100]#, 200, 500, 1000]
    DEGREE = 3
    GRAPHS_PER_SIZE = 10

    algorithms = {
        "PrioList": PriorityList.prio_list_output,
        "Super Trees": SuperTreesArborescenceDecomposition,
        "Beacon Nodes": beacon_nodes_algorithm,
        "Global Controller": global_controller_generator,
        "Simple Decomposition": GreedyArborescenceDecomposition
    }

    synthetic_graphs = generate_random_graphs(
        NODE_SIZES, degree=DEGREE, graphs_per_size=GRAPHS_PER_SIZE
    )

    results = run_all_algorithms_on_synthetic_graphs(
        algorithms,
        synthetic_graphs,
        failure_prob=0.2,
        sample_size=100,
        measure_depth=True
    )

    return results

# ✅ Now __main__ works fine
if __name__ == "__main__":
    NUM_RUNS = 8

    with Pool(cpu_count()) as pool:
        all_run_results = pool.map(one_run, range(NUM_RUNS))

    summarize_multiple_runs(all_run_results, output_dir="synthetic_plots", save_plots=True)
