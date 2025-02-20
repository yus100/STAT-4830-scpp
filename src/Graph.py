from uuid import uuid4


class Node:
    def __init__(self, id = None):
        if id is None:
            id = uuid4()
        self.id = id

    def __repr__(self):
        return f"Node({self.id})"
    
    # allow hashing
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

class Edge:
    def __init__(self, node1, node2, weight=1):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.permanent = False

    def __repr__(self):
        return f"Edge({self.node1}, {self.node2}, weight={self.weight})"
    
    # allow hashing
    def __hash__(self):
        return hash((self.node1, self.node2))

    def __eq__(self, other):
        return (self.node1, self.node2) == (other.node1, other.node2)

class Graph:
    def __init__(self):
        self.nodes = {}
        self.adjacency_list = {}
        self.edges: set = set()

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.adjacency_list[node.id] = []

    def add_edge(self, node1_id, node2_id, weight=1):
        if node1_id not in self.nodes or node2_id not in self.nodes:
            raise ValueError("Nodes must exist in the graph")
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        edge = Edge(node1, node2, weight)
        self.adjacency_list[node1_id].append((node2_id, edge))
        self.adjacency_list[node2_id].append((node1_id, edge)) # For undirected graph
        self.edges.add(edge)

    def remove_node(self, node_id):
        pass
    def remove_edge(self, node1_id, node2_id):
        pass

    def get_neighbors(self, node_id):
        if node_id not in self.nodes:
            raise ValueError("Node not in graph")
        return self.adjacency_list[node_id]

    def get_all_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def __repr__(self):
        return f"Graph(nodes={list(self.nodes.keys())}, adjacency_list={self.adjacency_list})"

    