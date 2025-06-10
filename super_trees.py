import networkx as nx
import random
import numpy as np
from priority_list import PriorityList
from arborescences import reset_arb_attribute, get_arborescence_list, TestCut
from heapq import heappush, heappop


def SuperTreesArborescenceDecomposition(g):
    reset_arb_attribute(g)
    gg = g.to_directed()
    K = g.graph['k']
    k = K
    prev_avg_failure = -1  # start with lowest possible

    while k > 0:
        T = FindSuperTrees(gg, k, prev_avg_failure)
        if T is None or len(T.edges()) == 0:
            #print(f"⚠️ No valid tree for k={k}. Restoring edges and retrying with k={k-1}...")
            gg = g.to_directed()
            k -= 1
            continue

        avg_failure = sum(g[u][v]['failure'] for u, v in T.edges()) / len(T.edges())
        if avg_failure <= prev_avg_failure:
            #print(f"⛔ Tree failure avg {avg_failure:.3f} not higher than previous {prev_avg_failure:.3f}")
            gg = g.to_directed()
            k -= 1
            continue

        prev_avg_failure = avg_failure
       # print(f"✅ Tree {K - k + 1} built with average failure {avg_failure}")
        
        for (u, v) in T.edges():
            g[u][v]['arb'] = K - k
        
        gg.remove_edges_from(T.edges())
        k -= 1

    return get_arborescence_list(g)

def lookahead_failure_score(g, node, depth_left, visited=None):
    if visited is None:
        visited = set()
    if depth_left == 0 or node in visited:
        return []

    visited.add(node)
    scores = []

    for neighbor in g.predecessors(node):
        if g.has_edge(neighbor, node):
            f = g[neighbor][node].get('failure', 1.0)
            scores.append(f)
            deeper = lookahead_failure_score(g, neighbor, depth_left - 1, visited.copy())
            scores.extend(deeper)

    return scores


def FindSuperTrees(g, k, min_avg_failure):
    T = nx.DiGraph()
    root = g.graph['root']
    lookahead_depth = g.graph.get('lookahead', 0)  # Default to 0 if not set
    T.add_node(root)
    R = {root}
    dist = {root: 0}
    h = []

    # Initialize with all predecessors to root, sorted by lookahead failure score
    preds = list(g.predecessors(root))
    tupPreds = []

    for p in preds:
        f = g[p][root].get('failure', float('inf'))
        if lookahead_depth > 0:
            scores = lookahead_failure_score(g, p, lookahead_depth)
            if scores:
                f = sum(scores) / len(scores)
        tupPreds.append((p, f))

    preds = [node for node, _ in sorted(tupPreds, key=lambda x: x[1])]

    for x in preds:
        failure = g[x][root].get('failure', float('inf'))
        if lookahead_depth > 0:
            scores = lookahead_failure_score(g, x, lookahead_depth)
            if scores:
                failure = sum(scores) / len(scores)
        heappush(h, (failure, 0, (x, root)))

    total_failure = 0
    edge_count = 0

    while h:
        failure_val, d, e = heappop(h)
        u, v = e

        if g.has_edge(u, v):
            g.remove_edge(u, v)
        else:
            continue

        if u not in R and (k == 1 or TestCut(g, u, g.graph['root']) >= k - 1):
            R.add(u)
            T.add_edge(u, v)
            dist[u] = d + 1
            total_failure += failure_val
            edge_count += 1

            avg_failure_so_far = total_failure / edge_count if edge_count > 0 else 0
            if avg_failure_so_far <= min_avg_failure:
                print(f"❌ Aborting tree: avg failure {avg_failure_so_far:.3f} ≤ min required {min_avg_failure:.3f}")
                return None

            new_preds = list(g.predecessors(u))
            tupPreds = []

            for p in new_preds:
                if p in R:
                    continue
                f = g[p][u].get('failure', float('inf'))
                if lookahead_depth > 0:
                    scores = lookahead_failure_score(g, p, lookahead_depth)
                    if scores:
                        f = sum(scores) / len(scores)
                tupPreds.append((p, f))

            new_preds = [node for node, _ in sorted(tupPreds, key=lambda x: x[1])]

            for x in new_preds:
                if x not in R:
                    failure = g[x][u].get('failure', float('inf'))
                    if lookahead_depth > 0:
                        scores = lookahead_failure_score(g, x, lookahead_depth)
                        if scores:
                            failure = sum(scores) / len(scores)
                    heappush(h, (failure, d + 1, (x, u)))
        else:
            g.add_edge(u, v)

    # Ensure root is included
    if g.graph['root'] not in T.nodes():
        print("Root wasn't in here")
        for node in T.nodes():
            if g.has_edge(node, root):
                T.add_edge(node, root)
                break

    return T if len(T.edges()) > 0 else None



# ---- CONFIG ----
NUM_GRAPHS = 100
NUM_NODES = 100
DEGREE = 3
MAX_LOOKAHEAD = 6

def run_lookahead_analysis_on_multiple_graphs():
    dl_per_graph = []

    for g_idx in range(NUM_GRAPHS):
        G = nx.random_regular_graph(DEGREE, NUM_NODES)
        G = G.to_directed()
        for u, v in G.edges():
            G[u][v]['failure'] = round(random.uniform(0.1, 0.9), 2)
        root = random.choice(list(G.nodes()))
        G.graph['root'] = root
        G.graph['k'] = 3
        dl_per_lookahead = []

        print(f"\n📊 Graph {g_idx+1}/{NUM_GRAPHS}")
        for lookahead in range(MAX_LOOKAHEAD + 1):
            print(f"🔍 Lookahead {lookahead}")
            G.graph['lookahead'] = lookahead
            trees = SuperTreesArborescenceDecomposition(G)
            dls = PriorityList.danger_level_all_trees(trees, G)
            avg_dl = np.mean(dls)
            print(f"🌡️  DL = {avg_dl:.4f}")
            dl_per_lookahead.append(avg_dl)

        dl_per_graph.append(dl_per_lookahead)

    return dl_per_graph
def plot_lookahead_results(dl_per_graph):
    import numpy as np
    import matplotlib.pyplot as plt

    x = list(range(MAX_LOOKAHEAD + 1))
    dl_array = np.array(dl_per_graph)
    mean_dl = np.mean(dl_array, axis=0)
    std_dl = np.std(dl_array, axis=0)

    # Plot all individual graph lines in light gray
    for dl_series in dl_per_graph:
        plt.plot(x, dl_series, color='lightgray', alpha=0.4)

    # Plot average with error bars
    plt.errorbar(
        x, mean_dl, yerr=std_dl, fmt='-o', color='black',
        linewidth=2.5, capsize=4
    )

    plt.xlabel("Lookahead")
    plt.ylabel("Average Danger Level")
    plt.title(f"DL vs Lookahead ({NUM_GRAPHS} d={DEGREE} regular graphs, n={NUM_NODES})")
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
# === Run all ===
if __name__ == "__main__":
    dl_results = run_lookahead_analysis_on_multiple_graphs()
    plot_lookahead_results(dl_results)