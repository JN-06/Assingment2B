import json
import os
import time
import networkx as nx
from graph import build_graph
from model import load_models, build_dynamic_graph
from search import ida_star

# Total travel time
def calculate_travel_time(G, path):
    total_time = 0.0
    for u, v in zip(path, path[1:]):
        edge = G.get_edge_data(u, v, {})
        total_time += float(
            edge.get("travel_time", edge.get("weight", 0.0)) or 0.0
        )
    return total_time

# Input validation
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

# Run system
def run_system(name, model, model_type, start, goal, y_scaler, scaler_X):
    G = build_dynamic_graph(model, model_type, y_scaler, scaler_X)
    print(f"\nRunning {name}...")
    coords = nx.get_node_attributes(G, "pos")
    start_time = time.time()
    path = ida_star(G, coords, start, [goal])
    execution_time = time.time() - start_time

    if not path:
        print(f"❌ {name}: No path found")
        return None, execution_time, G

    print(f"\n✔ {name}")
    print("Path:", " -> ".join(map(str, path)))
    print("Nodes:", len(path))
    print("Estimated Travel Time:", f"{calculate_travel_time(G, path):.2f}", "minutes")
    print("Execution Time:", f"{execution_time:.6f}", "seconds")
    print("-" * 40)

    return path, execution_time, G


# Main program
def main():
    print("\n====================================")
    print(" A2B TRAFFIC FLOW PREDICTION")
    print(" LSTM vs GRU vs Decision Tree accuracy with IDA* travel time")
    print("====================================\n")

    print("Loading models...")
    lstm, gru, dt, y_scaler, scaler_X = load_models()
    print("✔ Models loaded")

    base_graph = build_graph()
    start = get_valid_node("Enter Origin SCATS ID: ", base_graph)
    goal = get_valid_node("Enter Destination SCATS ID: ", base_graph)

    run_system("LSTM + IDA*", lstm, "LSTM", start, goal, y_scaler, scaler_X)
    run_system("GRU + IDA*", gru, "GRU", start, goal, y_scaler, scaler_X)
    run_system("DT + IDA*", dt, "DT", start, goal, y_scaler, scaler_X)

    # load train result
    results_file = "model/results.json"

    if not os.path.exists(results_file):
        print("\n❌ results.json not found. Run train.py first.")
        return
    with open(results_file, "r") as f:
        results_dict = json.load(f)
    results = list(results_dict.items())
    best = max(results, key=lambda x: x[1])

    print("\n====================================")
    print(" FINAL COMPARISON")
    print("====================================")
    print("Best Model:", best[0])
    print("Accuracy:", f"{best[1]:.2f}%")
    print("\n====================================")

if __name__ == "__main__":
    main()