from graph import build_graph
from model import load_models, build_dynamic_graph
from search import ida_star
import networkx as nx


# =========================
# INPUT VALIDATION
# =========================
def get_valid_node(prompt, G):
    while True:
        try:
            node = int(input(prompt))

            if node in G.nodes():
                return node
            else:
                print("❌ Invalid SCATS ID. Please try again.\n")

        except ValueError:
            print("❌ Please enter a valid number.\n")


# =========================
# COST CALCULATION
# =========================
def calculate_cost(G, path):
    if not path:
        return None

    cost = 0
    for i in range(len(path) - 1):
        cost += G[path[i]][path[i + 1]]["weight"]
    return cost


# =========================
# RUN SYSTEM
# =========================
def run_system(name, model, model_type, start, goal):
    G = build_dynamic_graph(model, model_type)
    print(f"\nRunning {name}...")

    # Get coords and pass goal as list
    coords = nx.get_node_attributes(G, "pos")
    path = ida_star(G, coords, start, [goal])

    if not path:
        print(f"❌ {name}: No path found")
        return None, None, G

    cost = calculate_cost(G, path)

    print(f"\n✔ {name}")
    print("Path:", " -> ".join(map(str, path)))
    print("Cost:", round(cost, 4))
    print("Nodes:", len(path))
    print("-" * 40)

    return path, cost, G


# =========================
# MAIN PROGRAM
# =========================
def main():
    print("\n====================================")
    print(" A2B TRAFFIC ROUTE SYSTEM")
    print(" LSTM vs GRU vs Decision Tree (IDA*)")
    print("====================================\n")

    print("Loading models...")
    lstm, gru, dt, y_scalar = load_models()
    print("✔ Models loaded")

    # =========================
    # BUILD BASE GRAPH FOR VALIDATION
    # =========================
    base_graph = build_graph()

    # =========================
    # SAFE USER INPUT
    # =========================
    start = get_valid_node("Enter Origin SCATS ID: ", base_graph)
    goal = get_valid_node("Enter Destination SCATS ID: ", base_graph)

    # =========================
    # RUN ALL MODELS
    # =========================
    lstm_path, lstm_cost, _ = run_system("LSTM + IDA*", lstm, "LSTM", start, goal)
    gru_path, gru_cost, _ = run_system("GRU + IDA*", gru, "GRU", start, goal)
    dt_path, dt_cost, _ = run_system("DT + IDA*", dt, "DT", start, goal)

    # =========================
    # FINAL COMPARISON
    # =========================
    print("\n====================================")
    print(" FINAL COMPARISON")
    print("====================================")

    results = [
        ("LSTM", lstm_cost),
        ("GRU", gru_cost),
        ("DT", dt_cost)
    ]

    results = [r for r in results if r[1] is not None]

    if results:
        # Sort by cost, then by name for consistent tie-breaking
        results_sorted = sorted(results, key=lambda x: (x[1], x[0]))
        best = results_sorted[0]
    
        # Check for ties
        tied = [r for r in results if abs(r[1] - best[1]) < 0.0001]
    
        if len(tied) > 1:
            # Multiple winners
            winners = ", ".join([r[0] for r in tied])
            print(f"Best Model:  {winners}")
            print(f"Lowest Cost:  {best[1]:.2f}")
            print(f"\nNote: {len(tied)} models tied for best cost!")
        else:
            # Single winner
            print(f"Best Model:  {best[0]}")
            print(f"Lowest Cost:  {best[1]:.2f}")
    else:
        print("\n❌ No valid paths found")

    print("\n====================================")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()