import networkx as nx
import random
from collections import defaultdict

class MyGraph:
    def __init__(self, filepath=None, num_nodes=None, num_topics=3):
        self.G = nx.DiGraph()
        self.num_topics = num_topics
        
        if filepath:
            self.load_graph(filepath)
        elif num_nodes:
            self.generate_graph(num_nodes)
        else:
            raise ValueError("Either 'filepath' or 'num_nodes' must be provided.")
        
        self.initialize_parameters()

    def load_graph(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith("#"):
                    continue
                u, v = map(int, line.strip().split('\t'))
                self.G.add_edge(u, v)

    def generate_graph(self, num_nodes):
        for i in range(num_nodes):
            self.G.add_node(i)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() > 0.7:
                    self.G.add_edge(i, j)
                if random.random() > 0.7:
                    self.G.add_edge(j, i)
                    
        # Randomly set state for 1/3 of the nodes
        random_nodes = random.sample(list(self.G.nodes()), num_nodes // 3)
        for node in random_nodes:
            self.G.nodes[node]['state'] = {f'topic_{random.randint(1, self.num_topics)}'}

    def initialize_parameters(self):
        for node in self.G.nodes():
            self.G.nodes[node]['c'] = 1
            self.G.nodes[node]['state'] = self.G.nodes[node].get('state', set())
            self.G.nodes[node]['p'] = {f'topic_{i + 1}': random.uniform(0.1, 1) for i in range(self.num_topics)}
            self.G.nodes[node]['gamma'] = {f'topic_{i + 1}': random.uniform(0.1, 1) for i in range(self.num_topics)}
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = 1 / self.G.in_degree(v)
