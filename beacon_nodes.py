from platform import node
import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.readwrite.graphml import read_graphml
from arborescences import GreedyArborescenceDecomposition, Network
from graph_info_printer_and_utils import GraphUtils
# Load a sample topology from Topology Zoo
# Ensure you have a GraphML file from Topology Zoo


def beacon_nodes_finder(G, k = 4):
    """
    Goes through nodes and sort them based on their degrees and the accumulative degrees of the neighboring nodes as well

    Parameters: 

    G is the graph that will be handled
    k defines the number of beacon nodes that shall be handled

    return:
    beacon nodes in list
    """
    K = k  
    node_deg_infos = []
    for node in G.nodes:    
        neighbors = G.neighbors(node)
        neighbors_score = sum(G.degree[nei] for nei in neighbors)           
        node_deg_infos.append((node, G.degree[node], neighbors_score))

    sorted_nodes = sorted(node_deg_infos, key=lambda x: (x[1], x[2]), reverse=True)

    return  [node for node,_,_ in sorted_nodes[:k]]


def beacon_nodes_generator(G, beacons): 
    target = G.graph['root']
    for beacon in beacons: 
        try:
            path = nx.shortest_path(G, source=beacon, target=target, weight=lambda u, v, d: d['failure'])
        except nx.NetworkXNoPath:
            continue 
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            if 'PathToRoot' not in G.nodes[node]:
                G.nodes[node]['PathToRoot'] = next_node     
    return G


def beacon_nodes_algorithm(G, k=4):
    """
    Assigns 'PathToRoot' attributes to nodes in the provided trees based on
    paths from selected beacon nodes to the root. PathToRoot points toward
    the root in the global graph G, even if that node lies outside the tree.

    Parameters:
        G (networkx.Graph): The full graph.
        trees (list): List of directed trees (DiGraphs) generated externally.
        k (int): Number of beacon nodes to select.

    Returns:
        list: List of updated trees with 'PathToRoot' attributes assigned.
    """
    trees = GreedyArborescenceDecomposition(G)
    target = G.graph['root']
    beacons = beacon_nodes_finder(G, k)

    # Precompute all beacon-to-root paths in the global graph
    paths_to_root_global = {}
    for beacon in beacons:
        try:
            path = nx.shortest_path(G, source=beacon, target=target, weight=lambda u, v, d: d['failure'])
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if u not in paths_to_root_global:
                    paths_to_root_global[u] = v  # Next hop toward root in global view
        except nx.NetworkXNoPath:
            continue

    # Assign PathToRoot within each tree — but only if the edge is present in the tree
    for tree in trees:
        for node in tree.nodes():
            if node in paths_to_root_global:
                next_hop = paths_to_root_global[node]
                if tree.has_edge(node, next_hop):  # Valid edge in this tree
                    tree.nodes[node]['PathToRoot'] = next_hop
                else:
                    tree.nodes[node]['PathToRoot'] = None
            elif node == target:
                tree.nodes[node]['PathToRoot'] = None
            else:
                tree.nodes[node]['PathToRoot'] = None

    return trees


#################################################################################### 
if __name__ == "__main__":
    TOPOLOGY_FILE = "Internode.graphml"
    G = read_graphml(TOPOLOGY_FILE)
    G = G.to_undirected()
    edg = list(G.edges())
    nod = list(G.nodes())
    fails = random.sample(edg, 5)
    G.graph['root'] = random.sample(nod, 1)[0]
    GraphUtils.assign_random_survivability(G)
    beacons =beacon_nodes_finder(G, 4)
    G = beacon_nodes_generator(G, beacons)
    print(f"The beacon nodes are {beacons} and the root is {G.graph['root']}")
    for node, data in G.nodes(data=True):
        if 'PathToRoot' in data:
            print(f"Node {node} → PathToRoot: {data['PathToRoot']}")        #this only prints the nodes with pathtoRoot info to them

    #for node, data in G.nodes(data=True):
    #    print(f"Node {node}: {data}")       #prints alNl nodes info