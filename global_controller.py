from beacon_nodes import beacon_nodes_algorithm
# Load a sample topology from Topology Zoo
# Ensure you have a GraphML file from Topology Zoo

def global_controller_generator(G): 
    V = list(G.nodes())
    return  beacon_nodes_algorithm(G, len(V))

"""
if __name__ == "__main__":

    TOPOLOGY_FILE = "Internode.graphml"
    G = read_graphml(TOPOLOGY_FILE)
    G = G.to_undirected()
    edg = list(G.edges())
    nod = list(G.nodes())
    fails = random.sample(edg, 5)
    G.graph['root'] = random.sample(nod, 1)[0]
    GraphUtils.assign_random_survivability(G)
    T = global_controller_generator(G)
    print(f"The root is {G.graph['root']}")
    for node, data in T.nodes(data=True):
        print(f"Node {node}: {data}")       #prints all nodes info
        """