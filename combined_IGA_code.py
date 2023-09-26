
import networkx as nx
import random
from collections import defaultdict
from Graph_net import MyGraph

# Function to simulate the spread of misinformation based on the MT-LT model
def simulate_spread_MT_LT_updated_G(G, sources):
    activated_nodes_by_topic = {topic: set() for topic in sources.keys()}
    
    for topic, source_list in sources.items():
        for source in source_list:
            if source in G.nodes:
                G.nodes[source]['state'].add(topic)  # Activate the source nodes for the given topic
                activated_nodes_by_topic[topic].add(source)
    
    while True:
        new_activations = set()
        
        for topic, activated_nodes in activated_nodes_by_topic.items():
            for source in list(activated_nodes):
                if source not in G.nodes:
                    continue
                
                for neighbor in G.successors(source):
                    if topic not in G.nodes[neighbor]['state']:
                        
                        # Calculate the total influence on 'neighbor' for the current topic
                        influence_sum = sum(
                            G[source][neighbor]['weight'] * G.nodes[u]['p'].get(topic, 0)
                            for u in activated_nodes
                        )
                        
                        # Check if the influence is enough to activate 'neighbor'
                        if influence_sum >= G.nodes[neighbor]['gamma'].get(topic, 0):
                            new_activations.add((neighbor, topic))
                            G.nodes[neighbor]['state'].add(topic)
                            
        if not new_activations:
            break
        
        for node, topic in new_activations:
            activated_nodes_by_topic[topic].add(node)
            
    num_activated_nodes_by_topic = {topic: len(nodes) for topic, nodes in activated_nodes_by_topic.items()}
    return G, num_activated_nodes_by_topic

def separate_graphs_by_topic(G, topics):
    separate_graphs = {}
    
    for topic in topics:
        # Create a new directed graph for this topic
        G_topic = G.copy()
        
        for u, data in G.nodes(data=True):
            if topic not in data['state']:
                # Set state to empty set if the node is not activated by this topic
                G_topic.nodes[u]['state'] = set()
                
        separate_graphs[topic] = G_topic

    return separate_graphs

def unify_source_nodes_safe(Gi, Si , topic):
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





# Monte Carlo simulation with subgraph sampling
def monte_carlo_simulation_MT_LT_subgraph(G, source_sets, T=10, subgraph_ratio=0.7):
    count = 0
    for _ in range(T):
        sampled_nodes = random.sample(G.nodes(), int(len(G.nodes()) * subgraph_ratio))
        subG = G.subgraph(sampled_nodes)
        sub_source_sets = {topic: [s for s in sources if s in subG.nodes] for topic, sources in source_sets.items()}
        
        for topic, sources in sub_source_sets.items():
            Ni = simulate_spread_MT_LT_updated_G(subG, sub_source_sets, topic)
            count += Ni
    return count / (T * len(source_sets))

# IGA Algorithm with MT-LT model and subgraph sampling
# Initialize the graph



def f_dfs_iterative(tree, root):
    if not tree.has_node(root):
        return 0
    
    visited = set()
    stack = [(root, iter(tree.successors(root)))]
    r = 0
    
    while stack:
        parent, children = stack[-1]
        try:
            child = next(children)
            if child not in visited:
                r += 1
                visited.add(child)
                stack.append((child, iter(tree.successors(child))))
        except StopIteration:
            stack.pop()
    
    return r
def calculate_f(Ti, u):
    if Ti.out_degree(u) == 0:  # u is a leaf node
        return 1
    r = 1  # Initialize r
    for v in Ti.successors(u):  # v is a child of u
        r += calculate_f(Ti, v)
    return r

def generate_live_edge_samples(Gi_prime, num_samples):
    live_edge_samples = []
    for _ in range(num_samples):
        sample = nx.DiGraph()
        for u, v, data in Gi_prime.edges(data=True):
            if random.random() <= data.get('weight', 1):
                sample.add_edge(u, v, weight=data.get('weight', 1))
        live_edge_samples.append(sample)
    return live_edge_samples

# Define the function to generate trees rooted at Hi from a live-edge sample graph
def generate_trees_from_graph(sample, Hi):
    trees = []
    
    if Hi not in sample.nodes():
        return trees  # Return empty list if Hi is not in sample
    
    for node in nx.dfs_preorder_nodes(sample, Hi):
        if node == Hi:
            tree = nx.DiGraph()
            tree.add_node(Hi)
        else:
            for parent in sample.predecessors(node):
                if parent == Hi:
                    trees.append(tree)  # Add the tree only when Hi has successors
                if parent in tree.nodes():
                    tree.add_edge(parent, node)
                    
    return [tree for tree in trees if tree.number_of_nodes() > 1]  # Only return trees that have more than just Hi

def update_f_values(Ti, u, f_values):
    # Remove node u from Ti if it exists and update f_values
    if u in Ti:
        descendants = list(nx.descendants(Ti, u))
        
        # Remove u and all its descendants
        Ti.remove_nodes_from([u] + descendants)
        
    for v in Ti.nodes():
        if v in f_values:
            # If v is a prefix of u, update its f-value
            if u in nx.descendants(Ti, v):
                f_values[Ti][v] = f_values[Ti][v] - f_values[Ti].get(u , 0)
            else:
                # Otherwise, simply calculate the new f-value
                f_values[Ti][v] = calculate_f(Ti, v)

def calculate_delta(u, Ti_sets , f_values):
    delta_u = 0
    q = len(Ti_sets)  # Number of topics
    for topic in Ti_sets:
        for Ti_list in Ti_sets[topic]:
            for Ti in Ti_list:
                Ti_copy = Ti.copy()
                if u in Ti.nodes(): 
                    Ti_copy.remove_node(u)
                    
                    delta_u += (1/q) * (f_values[Ti]['Hi'] - calculate_f(Ti_copy , 'Hi'))
    return delta_u

def GEA(G, B, num_samples=10):
    U = set(G.nodes())
    A1 = set()
    topics = ['topic_1', 'topic_2', 'topic_3']
    q = len(topics)
    
    # Step 2 and 3: Build Gi and Merge Gi
    separated_graphs = separate_graphs_by_topic(G, topics)
    source_nodes_by_topic = {}

    # Duyệt qua từng đồ thị đã được tách và tìm các đỉnh nguồn
    for topic, G_topic in separated_graphs.items():
        source_nodes = [node for node, data in G_topic.nodes(data=True) if data['state'] == {topic}]
        source_nodes_by_topic[topic] = source_nodes

    unified_graphs = {}
    for topic in topics:
        unified_graphs[topic], Hi = unify_source_nodes_safe(separated_graphs[topic], source_nodes_by_topic[topic] , topic)
    
    Ti_sets = {}
    

    for topic, Gi_prime in unified_graphs.items():
        Hi = 'Hi'
        sample_graphs = generate_live_edge_samples(Gi_prime, num_samples)
        
        Ti_sets[topic] = [generate_trees_from_graph(sample, Hi) for sample in sample_graphs]
    
    # Initialize sigma_hat (approximated influence spread) for each node to 0
    sigma_hat = {node: 0 for node in list(G.nodes()) + ['Hi']}
    f_values = {}
    # Step 6: Calculate sigma_hat for all u in THi by Algorithm 4
    for topic in Ti_sets:
        for Ti_list in Ti_sets[topic]:
            for Ti in Ti_list:
                f_values[Ti] = {}

                for u in Ti.nodes():
                    f_values[Ti][u] = calculate_f(Ti, u)
                    sigma_hat[u] += f_values[Ti][u]
    
    # Normalize sigma_hat
    for u in sigma_hat:
        sigma_hat[u] = sigma_hat[u] / (q * num_samples)
    
    # Step 8: Find umax
    umax = max((node for node in sigma_hat.keys() if node != 'Hi' and G.nodes[node]['c'] <= B), key=lambda node: sigma_hat[node])
    print(umax)
    count = 0
    # Step 9: Repeat
    while U:
        # Step 10: Find cmin
        cmin = min(G.nodes[node]['c'] for node in U)
        
        # Step 11: Check budget
        if cmin + sum(G.nodes[node]['c'] for node in A1) > B:
            break
        
        # Step 12: Find u with max delta(A1, u)
        # u = max(U, key=lambda node: sigma_hat[node] - sum(sigma_hat[v] for v in A1 ))
        u = max(U, key=lambda node: calculate_delta(node , Ti_sets, f_values))
        print(u)
        U.remove(u)
        # Step 14: Check budget again
        if sum(G.nodes[node]['c'] for node in A1) + G.nodes[u]['c'] <= B:
            A1.add(u)
            for topic in Ti_sets:
                for Ti_list in Ti_sets[topic]:
                    for Ti in Ti_list:
                        if u in Ti:
                        # Block node u and update f(Ti, v)
                        # (Assuming you have an update_f_values function)
                            update_f_values(Ti, u, f_values)

        
       
    
    # Step 23: Finalize the set of nodes A
    sigma_hat_A1 = sum(sigma_hat[u] for u in G.nodes() if u not in A1)
    A = A1 if sigma_hat_A1 > sigma_hat[umax] else {umax}
    print(sigma_hat_A1)

    
    return A








filepath = 'C:\\Users\\Admin\\Downloads\\data\\p2p-Gnutella08.txt'
num_topics = 3

my_graph = MyGraph(filepath=filepath, num_topics=3)

# Truy cập đồ thị G
G = my_graph.G

budget = 20
T_reduced = 1  # Reduced number of Monte Carlo simulations
subgraph_ratio_reduced = 0.3  # Reduced ratio of nodes to keep in the subgraph
activated_graphs = {}

source_sets = {f'topic_{i+1}': random.sample(list(G.nodes()), min(100, len(G.nodes()))) for i in range(num_topics)}

print(source_sets)

updated_G, num_activated_nodes_by_topic = simulate_spread_MT_LT_updated_G(G.copy(), source_sets)

# Check the updated states for a subset of nodes
sample_nodes = random.sample(list(updated_G.nodes()), min(10, len(G.nodes())))
print({node: G.nodes[node]['state'] for node in sample_nodes})
print(num_activated_nodes_by_topic)
gea = GEA(updated_G, B = 20, num_samples=100)
print(gea)