import math

# Heuristic
def heuristic(a, b, coords):
    x1, y1 = coords[a]
    x2, y2 = coords[b]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def min_heuristic(node, goals, coords):
    return min(heuristic(node, g, coords) for g in goals)

# Path reconstuction
def reconstruct_path(parent, start, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return list(reversed(path))

# IDA* for least-moves search
def ida_star(graph, coords, start, goals):
    def search(node, g, threshold, parent, visited):
        h = min_heuristic(node, goals, coords)
        f = g + h
        if f > threshold:
            return f
        if node in goals:
            return reconstruct_path(parent, start, node)
        min_threshold = float("inf")

        for neighbor in graph[node]:
            if neighbor in visited:
                continue
            edge = graph.get_edge_data(node, neighbor, {})
            cost = float(edge.get("weight", 1.0) or 1.0)
            visited.add(neighbor)
            parent[neighbor] = node
            temp = search(
                neighbor,
                g + cost,
                threshold,
                parent,
                visited
            )
            if isinstance(temp, list):
                return temp
            min_threshold = min(min_threshold, temp)
            visited.remove(neighbor)
            parent.pop(neighbor, None)
        return min_threshold
    threshold = min_heuristic(start, goals, coords)

    while True:
        visited = {start}
        parent = {start: None}
        temp = search(start, 0, threshold, parent, visited)
        if isinstance(temp, list):
            return temp
        if temp == float("inf"):
            return None
        threshold = temp