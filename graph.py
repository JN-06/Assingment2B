import networkx as nx
import math

# Nodes
NODES = [
    3120, 3122, 3126, 3180,
    4030, 4032, 4034, 4035,
    4040, 4043
]

# Edges
EDGES = [
    (3120, 3122),
    (3120, 4030),
    (3120, 3180),
    (3122, 3126),
    (3122, 4034),
    (3126, 4032),
    (4032, 4043),
    (4030, 4034),
    (4034, 4043),
    (4040, 4034),
    (4040, 4043),
    (4035, 4040),
    (3180, 4040),
    (3180, 4030),
    (3180, 4035),
    # Reverse paths
    (4043, 4032),
    (4043, 4034),
    (4043, 4040),
    (4032, 3126),
    (3126, 3122),
    (3122, 3120),
    (4034, 3122),
    (4034, 4030),
    (4034, 4040),
    (4030, 3120),
    (4030, 3180),
    (4040, 3180),
    (4040, 4035),
    (4035, 3180),
    (3180, 3120),
]

# Build graph
def build_graph():
    G = nx.DiGraph()
    positions = {
        3120: (0, 2),
        3122: (1, 2),
        3126: (2, 2),
        3180: (0, 0),
        4030: (1, 1),
        4032: (3, 2),
        4034: (2, 1),
        4035: (1, -1),
        4040: (2, 0),
        4043: (3, 1)
    }

    # add node
    for node in NODES:
        G.add_node(node, pos=positions[node])

    # ADD EDGES (FIXED)
    for u, v in EDGES:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        G.add_edge(
            u,
            v,
            distance=round(dist, 2),
            travel_time=0,
            weight=0
        )
    return G

# Test
if __name__ == "__main__":
    G = build_graph()
    print("Nodes:")
    print(G.nodes())
    print("\nEdges:")
    for edge in G.edges(data=True):
        print(edge)