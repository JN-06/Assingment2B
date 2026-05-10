import networkx as nx
import matplotlib.pyplot as plt

# =========================
# NODES (SCATS IDS)
# =========================
NODES = [
    3120, 3122, 3126, 3180,
    4030, 4032, 4034, 4035,
    4040, 4043
]

# =========================
# EDGES (TRAFFIC LINKS)
# =========================
EDGES = [
    (3120, 3122),
    (3122, 3126),
    (3126, 3180),
    (3180, 4030),
    (4030, 4032),
    (4032, 4034),
    (4034, 4035),
    (4035, 4040),
    (4040, 4043),

    # shortcuts
    (3120, 4034),
    (3126, 4030),
    (4030, 4034),
    (4032, 4040),
]

# =========================
# BUILD GRAPH
# =========================
def build_graph():
    G = nx.DiGraph()

    for n in NODES:
        G.add_node(n)

    for u, v in EDGES:
        G.add_edge(u, v)

    return G


# =========================
# DRAW GRAPH (CLEAN VERSION)
# =========================
def draw_graph(G):
    plt.figure(figsize=(12, 7))

    # FIXED LAYOUT (more structured than spring_layout)
    pos = nx.kamada_kawai_layout(G)

    # nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#4FC3F7",
        node_size=1600,
        edgecolors="black",
        linewidths=1.5
    )

    # edges (more visible like roads)
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="#555555",
        width=2
    )

    # labels (cleaner)
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight="bold",
        font_color="black"
    )

    plt.title("🚦 SCATS Traffic Network (Improved View)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================
# RUN FILE DIRECTLY
# =========================
if __name__ == "__main__":
    G = build_graph()
    print("Graph built successfully!")
    draw_graph(G)