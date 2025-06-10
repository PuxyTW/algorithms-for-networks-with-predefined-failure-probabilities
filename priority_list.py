
from arborescences import GreedyArborescenceDecomposition

class PriorityList():
    def danger_Level_path(G, path):
        DL = 0
        prev_nd = '' 
        for nd in path: 
            if prev_nd == '':
                DL=DL
                prev_nd = nd
            else: 
                DL += G[prev_nd][nd]['failure']
                prev_nd = nd
        return DL/(len(path)-1)

    def danger_level_tree(T, G): 
        DL = 0
        DL =sum(G[u][v]['failure'] for u, v in T.edges()) /len(T.edges())
        return DL
    
    def danger_level_all_trees(Trees, G): 
        DL = [0]*len(Trees)
        for i, t in enumerate(Trees): 
             DL[i] =sum(G[u][v]['failure'] for u, v in t.edges()) / len(t.edges())
        print(DL)
        return DL
    
    def priority_list_generator(DL):
        #turns it into a tuple with the index near the values
        return [index for index, _ in sorted(enumerate(DL), key=lambda x: x[1])] 

    def prio_list_output(G):
        Trees = GreedyArborescenceDecomposition(G)    
       # Calculate danger levels
        danger_levels = []
        for t in Trees:
            edges = list(t.edges())
            if edges:
                dl = sum(G[u][v]['failure'] for u, v in edges) / len(edges)
            else:
                dl = 0  # handle tree with no edges
            danger_levels.append(dl)
        
        # Get priority indices (sorted by danger level)
        priority_indices = sorted(range(len(Trees)), key=lambda i: danger_levels[i])
        
        # Return the trees ordered by priority
        sorted_trees = [Trees[i] for i in priority_indices]
        return sorted_trees

"""
#now just do the usual routing and send them sequentially through the prio list created 
if __name__ == "__main__":

    #PriorityList.danger_level_all_trees([1,2,3,5,7], "x")
    TOPOLOGY_FILE = "Internode.graphml"
    G = read_graphml(TOPOLOGY_FILE)
    G = G.to_undirected()
    edg = list(G.edges())
    nod = list(G.nodes())
    fails = random.sample(edg, 5)
    G.graph['k'] = 3
    G.graph['root'] = random.sample(nod, 1)[0]
    G.graph['fails'] = fails
    GraphUtils.assign_random_survivability(G)
    T1 = PriorityList.prio_list_output(G)
    print(f"Number of trees returned: {len(T1)}")   # ADD THIS LINE
    GraphPrints.print_trees_info(T1, G)
"""
