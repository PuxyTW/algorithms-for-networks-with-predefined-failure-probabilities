import networkx as nx
import random
import os
import matplotlib.pyplot as plt
from networkx.readwrite.graphml import read_graphml
from arborescences import GreedyArborescenceDecomposition, Network

def danger_level_tree(T, G): 
        DL = 0
        DL =sum(G[u][v]['failure'] for u, v in T.edges()) /len(T.edges())
        return DL

class GraphPrints(): 

    def print_trees_info(Trees, G):
        for i, t in enumerate(Trees): 
            print(f"Arborescence nr {i}")
            print(t)
            print(rf"The root is {t.graph['root']} and the danger level is {danger_level_tree(t, G)}")
            print(rf"Nodes: {t.nodes()}")
            print(rf"Edges: {t.edges()}")

    
    

class GraphUtils():
    
    def assign_random_survivability(G, min = 0.1, max=0.9):
        for u, v in G.edges():
            G[u][v]['failure'] = round(random.uniform(min, max), 2)  

    def most_survivable_path(G, source, target):
        path = nx.shortest_path(G, source=source, target=target, weight=lambda u, v, d: d['failure'])
        return path

    def simulate_edge_failure(G, edge):
        G_temp = G.copy()
        #counter = sth
        #for u,v in G_temp.edges():
        #    G[u][v]['failure']#roll the dice and then remove 
        #    G_temp.remove_edge(G[u][v])

        #TODO based on the probabilities, roll a dice on whether the edge should be removed or not
        #Once a edge is removed, decrement an inside counter and do it until it reaches 0.
        G_temp.remove_edge(*edge)
        is_connected = nx.is_connected(G_temp)
        return is_connected
        

    def assign_specific_failure_probabilities(G, params, graph_name="graph"):
        root = G.graph.get('root')
        if root is None:
            raise ValueError("Graph metadata does not contain a 'root' node (G.graph['root'] is missing).")

        leaves = [n for n in G.nodes() if G.degree(n) == 1]

        for u, v in G.edges():

            if (u == root or v == root) and params.get("failure_around_root") is not None:
                G[u][v]['failure'] = params["failure_around_root"]

            elif (u in leaves or v in leaves) and params.get("failure_on_leaves") is not None:
                G[u][v]['failure'] = params["failure_on_leaves"]

            elif params.get("failure_otherwise") is not None:
                G[u][v]['failure'] = params["failure_otherwise"]

            elif params.get("failure_in_range_min") is not None and params.get("failure_in_range_max") is not None:
                min_val = params["failure_in_range_min"]
                max_val = params["failure_in_range_max"]
                G[u][v]['failure'] = round(random.uniform(min_val, max_val), 2)

            else:
                # No applicable parameter, leave unchanged (or optionally raise warning/log)
                pass


                # Convert graph attributes
        for k, v in G.graph.items():
            if isinstance(v, list):
                G.graph[k] = str(v)

        # Convert node attributes
        for n, data in G.nodes(data=True):
            for attr, value in data.items():
                if isinstance(value, list):
                    G.nodes[n][attr] = str(value)

        # Convert edge attributes
        for u, v, data in G.edges(data=True):
            for attr, value in data.items():
                if isinstance(value, list):
                    G[u][v][attr] = str(value)
        # Save the altered graph
        altered_filename = f"{graph_name}_altered.graphml"
        os.makedirs("altered_graphs", exist_ok=True)
        nx.write_graphml(G, os.path.join("altered_graphs", altered_filename))

        return G