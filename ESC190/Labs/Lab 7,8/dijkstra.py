import numpy as np


class Node:
    def __init__(self, value):
        self.value = value
        self.connections = []
        self.distance_from_start = np.inf

class Con:
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight

def dijkstra(start, end):
    start.distance_from_start = 0
    visited = set([start])
    current = start
    while current != end:
        
        cur_dist = np.inf
        cur_v = None
        for node in visited:
            for con in node.connections:
                if con.node in visited:
                    continue
                if cur_dist > node.distance_from_start + con.weight:
                    cur_dist = node.distance_from_start + con.weight
                    cur_v = con.node
    
        current = cur_v
        current.distance_from_start = cur_dist
        visited.add(current)
    return current.distance_from_start
                                                
# Create the graph
node0 = Node(0)
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)

node0.connections = [Con(node1, 2), Con(node2, 4)]
node1.connections = [Con(node2, 1), Con(node3, 5)]
node2.connections = [Con(node3, 1), Con(node4, 3)]
node3.connections = [Con(node4, 2), Con(node5, 4)]
node4.connections = [Con(node5, 1)]


# Call the dijkstra function and print the shortest path
shortest_path = dijkstra(node0, node3)
print(shortest_path)

