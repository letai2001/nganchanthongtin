import networkx as nx
import random
from Graph_net import MyGraph

class MT_LT_GEA:
    def __init__(self, filepath, num_topics=3):
        self.my_graph = MyGraph(filepath=filepath, num_topics=num_topics)
        self.G = self.my_graph.G
        self.source_sets = {f'topic_{i+1}': random.sample(list(self.G.nodes()), min(100, len(self.G.nodes()))) for i in range(num_topics)}

    def simulate_spread_MT_LT_updated_G(self):
        G_copy = self.G.copy()
        activated_nodes_by_topic = {topic: set() for topic in self.source_sets.keys()}
    
        for topic, source_list in self.source_sets.items():
            for source in source_list:
                if source in self.G.nodes:
                    self.G.nodes[source]['state'].add(topic)  
                    activated_nodes_by_topic[topic].add(source)
        
        while True:
            new_activations = set()
            
            for topic, activated_nodes in activated_nodes_by_topic.items():
                for source in list(activated_nodes):
                    if source not in G_copy.nodes:
                        continue
                    
                    for neighbor in G_copy.successors(source):
                        if topic not in G_copy.nodes[neighbor]['state']:
                            
                            
                            influence_sum = sum(
                                G_copy[source][neighbor]['weight'] * G_copy.nodes[u]['p'].get(topic, 0)
                                for u in activated_nodes
                            )
                            
                            
                            if influence_sum >= G_copy.nodes[neighbor]['gamma'].get(topic, 0):
                                new_activations.add((neighbor, topic))
                                G_copy.nodes[neighbor]['state'].add(topic)
                                
            if not new_activations:
                break
            
            for node, topic in new_activations:
                activated_nodes_by_topic[topic].add(node)
                
        num_activated_nodes_by_topic = {topic: len(nodes) for topic, nodes in activated_nodes_by_topic.items()}
        return G_copy, num_activated_nodes_by_topic

    def separate_graphs_by_topic(self, G_update ,  topics):
        separate_graphs = {}
        
        for topic in topics:
            
            G_topic = G_update.copy()
            
            for u, data in G_update.nodes(data=True):
                if topic not in data['state']:
                   
                    G_topic.nodes[u]['state'] = set()
                    
            separate_graphs[topic] = G_topic

        return separate_graphs
        

    def unify_source_nodes_safe(self, Gi, Si, topic):
            Gi_prime = Gi.copy()
            Hi = 'Hi'
            Gi_prime.add_node(Hi)
            
            for x in Si:
                if x not in Gi_prime.nodes:
                    continue
                for v in list(Gi_prime.successors(x)):
                    if (Hi, v) not in Gi_prime.edges():
                        Gi_prime.add_edge(Hi, v)
                        Gi_prime[Hi][v]['weight'] = Gi_prime[x][v]['weight'] * Gi_prime.nodes[x]['p'].get(topic, 1)
                    else:
                        Gi_prime[Hi][v]['weight'] += Gi_prime[x][v]['weight'] * Gi_prime.nodes[x]['p'].get(topic, 1)
            
            Gi_prime.remove_nodes_from(Si)
            
            return Gi_prime, Hi



    def calculate_f(self, Ti, u):
        if Ti.out_degree(u) == 0:  # u is a leaf node
            return 1
        r = 1  # Initialize r
        for v in Ti.successors(u):  # v is a child of u
            r += self.calculate_f(Ti, v)
        return r

      

    def generate_live_edge_samples(self, Gi_prime, num_samples):
        live_edge_samples = []
        for _ in range(num_samples):
            sample = nx.DiGraph()
            for u, v, data in Gi_prime.edges(data=True):
                if random.random() <= data.get('weight', 1):
                    sample.add_edge(u, v, weight=data.get('weight', 1))
            live_edge_samples.append(sample)
        return live_edge_samples

       

    def generate_trees_from_graph(self, sample, Hi):
        trees = []
    
        if Hi not in sample.nodes():
            return trees  
        
        for node in nx.dfs_preorder_nodes(sample, Hi):
            if node == Hi:
                tree = nx.DiGraph()
                tree.add_node(Hi)
            else:
                for parent in sample.predecessors(node):
                    if parent == Hi:
                        trees.append(tree)  
                    if parent in tree.nodes():
                        tree.add_edge(parent, node)
                        
        return [tree for tree in trees if tree.number_of_nodes() > 1]  


    def update_f_values(self, Ti, u, f_values):
        if u in Ti:
            descendants = list(nx.descendants(Ti, u))
            
            Ti.remove_nodes_from([u] + descendants)
            
        for v in Ti.nodes():
            if v in f_values:
                if u in nx.descendants(Ti, v):
                    f_values[Ti][v] = f_values[Ti][v] - f_values[Ti].get(u , 0)
                else:
                    f_values[Ti][v] = self.calculate_f(Ti, v)


    def calculate_delta(self, u, Ti_sets, f_values):
        delta_u = 0
        q = len(Ti_sets)  # Number of topics
        for topic in Ti_sets:
            for Ti_list in Ti_sets[topic]:
                for Ti in Ti_list:
                    Ti_copy = Ti.copy()
                    if u in Ti.nodes(): 
                        Ti_copy.remove_node(u)
                        
                        delta_u += (1/q) * (f_values[Ti]['Hi'] - self.calculate_f(Ti_copy , 'Hi'))
        return delta_u


    def GEA(self, B, num_samples=10):
        G_update , num_activated_nodes_by_topic = self.simulate_spread_MT_LT_updated_G()
        sample_nodes = random.sample(list(G_update.nodes()), min(10, len(G_update.nodes())))
        print({node: G_update.nodes[node]['state'] for node in sample_nodes})
        print(num_activated_nodes_by_topic)

        U = set(G_update.nodes())
        A1 = set()
        topics = ['topic_1', 'topic_2', 'topic_3']
        q = len(topics)
        
        separated_graphs = self.separate_graphs_by_topic(G_update , topics)
        source_nodes_by_topic = {}

        for topic, G_topic in separated_graphs.items():
            source_nodes = [node for node, data in G_topic.nodes(data=True) if data['state'] == {topic}]
            source_nodes_by_topic[topic] = source_nodes

        unified_graphs = {}
        for topic in topics:
            unified_graphs[topic], Hi = self.unify_source_nodes_safe(separated_graphs[topic], source_nodes_by_topic[topic] , topic)
        
        Ti_sets = {}
        

        for topic, Gi_prime in unified_graphs.items():
            Hi = 'Hi'
            sample_graphs = self.generate_live_edge_samples(Gi_prime, num_samples)
            
            Ti_sets[topic] = [self.generate_trees_from_graph(sample, Hi) for sample in sample_graphs]
        
        sigma_hat = {node: 0 for node in list(self.G.nodes()) + ['Hi']}
        f_values = {}
        for topic in Ti_sets:
            for Ti_list in Ti_sets[topic]:
                for Ti in Ti_list:
                    f_values[Ti] = {}

                    for u in Ti.nodes():
                        f_values[Ti][u] = self.calculate_f(Ti, u)
                        sigma_hat[u] += f_values[Ti][u]
        
        for u in sigma_hat:
            sigma_hat[u] = sigma_hat[u] / (q * num_samples)
        
        umax = max((node for node in sigma_hat.keys() if node != 'Hi' and self.G.nodes[node]['c'] <= B), key=lambda node: sigma_hat[node])
        print(umax)
        count = 0
        while U:
            cmin = min(self.G.nodes[node]['c'] for node in U)
            
            if cmin + sum(self.G.nodes[node]['c'] for node in A1) > B:
                break
            
            u = max(U, key=lambda node: self.calculate_delta(node , Ti_sets, f_values))
            print(u)
            U.remove(u)
            if sum(self.G.nodes[node]['c'] for node in A1) + self.G.nodes[u]['c'] <= B:
                A1.add(u)
                for topic in Ti_sets:
                    for Ti_list in Ti_sets[topic]:
                        for Ti in Ti_list:
                            if u in Ti:
                                self.update_f_values(Ti, u, f_values)

            
        
        
        sigma_hat_A1 = sum(sigma_hat[u] for u in self.G.nodes() if u not in A1)
        A = A1 if sigma_hat_A1 > sigma_hat[umax] else {umax}
        print(sigma_hat_A1)

        
        return A



def main():
    filepath = 'C:\\Users\\Admin\\Downloads\\data\\p2p-Gnutella08.txt'
    num_topics = 3
    budget = 20

    model = MT_LT_GEA(filepath, num_topics)


    gea = model.GEA(budget, num_samples=25)
    print(gea)

if __name__ == "__main__":
    main()
