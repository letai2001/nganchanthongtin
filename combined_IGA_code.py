
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
        G_topic = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            if topic in G.nodes[u]['state'] and topic in G.nodes[v]['state']:
                # Add edge if both nodes are activated by this topic
                G_topic.add_edge(u, v, weight=data.get('weight', 1))
        separate_graphs[topic] = G_topic
    return separate_graphs

def unify_source_nodes_safe(Gi, Si):
    Gi_prime = Gi.copy()
    Hi = 'Hi'
    Gi_prime.add_node(Hi)
    
    for x in Si:
        if x not in Gi_prime.nodes:
            continue
        for v in list(Gi_prime.successors(x)):
            if (Hi, v) not in Gi_prime.edges():
                Gi_prime.add_edge(Hi, v)
                Gi_prime[Hi][v]['weight'] = Gi_prime[x][v]['weight'] * Gi_prime.nodes[x].get('p', 1)
            else:
                Gi_prime[Hi][v]['weight'] += Gi_prime[x][v]['weight'] * Gi_prime.nodes[x].get('p', 1)
    
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
    visited = set()
    if sample.number_of_edges() == 0:
        return trees
    for node in nx.dfs_preorder_nodes(sample, Hi):
        if node == Hi:
            tree = nx.DiGraph()
            tree.add_node(Hi)
            trees.append(tree)
        else:
            for parent in sample.predecessors(node):
                if parent in visited:
                    trees[-1].add_edge(parent, node)
        visited.add(node)
    return trees
def update_f_values(Ti, u, f_values):
    if u in Ti:
        Ti.remove_node(u)
    for v in Ti.nodes():
        f_values[v] = calculate_f(Ti, v)

def calculate_delta(u, Ti_sets , f_values):
    delta_u = 0
    q = len(Ti_sets)  # Number of topics
    for topic in Ti_sets:
        for Ti_list in Ti_sets[topic]:
            for Ti in Ti_list:
                if u in Ti.nodes():
                    delta_u += (1/q) * (calculate_f(Ti, 'Hi') - update_f_values(Ti, u , f_values))
    return delta_u

def GEA(G, B, num_samples=5):
    U = set(G.nodes())
    A1 = set()
    topics = ['topic_1', 'topic_2', 'topic_3']
    q = len(topics)
    
    # Step 2 and 3: Build Gi and Merge Gi
    separated_graphs = separate_graphs_by_topic(G, topics)
    source_nodes_by_topic = {topic: [node for node in G.nodes if topic in G.nodes[node]['state']] for topic in topics}
    unified_graphs = {}
    for topic in topics:
        unified_graphs[topic], Hi = unify_source_nodes_safe(separated_graphs[topic], source_nodes_by_topic[topic])
    
    Ti_sets = {}
    

    for topic, Gi_prime in unified_graphs.items():
        Hi = 'Hi'
        sample_graphs = generate_live_edge_samples(Gi_prime, num_samples)
        
        Ti_sets[topic] = [generate_trees_from_graph(sample, Hi) for sample in sample_graphs]
    
    # Initialize sigma_hat (approximated influence spread) for each node to 0
    sigma_hat = {node: 0 for node in G.nodes()}
    
    # Step 6: Calculate sigma_hat for all u in THi by Algorithm 4
    for topic in Ti_sets:
        for Ti_list in Ti_sets[topic]:
            for Ti in Ti_list:
                for u in Ti.nodes():
                    sigma_hat[u] += calculate_f(Ti, u)
    
    # Normalize sigma_hat
    for u in sigma_hat:
        sigma_hat[u] = sigma_hat[u] / (q * num_samples)
    
    # Step 8: Find umax
    umax = max((node for node in sigma_hat if G.nodes[node]['c'] <= B), key=lambda node: sigma_hat[node])
    f_values = {}
    # Step 9: Repeat
    while U:
        # Step 10: Find cmin
        cmin = min(G.nodes[node]['c'] for node in U)
        
        # Step 11: Check budget
        if cmin + sum(G.nodes[node]['c'] for node in A1) > B:
            break
        
        # Step 12: Find u with max delta(A1, u)
        # u = max(U, key=lambda node: sigma_hat[node] - sum(sigma_hat[v] for v in A1 ))
        u = max(U, key=lambda node: calculate_delta(node , Ti_sets, sigma_hat))
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
                            update_f_values(Ti, u, sigma_hat)

        
       
    
    # Step 23: Finalize the set of nodes A
    A = A1 if sigma_hat[max(A1, key=lambda node: sigma_hat[node])] > sigma_hat[umax] else {umax}
    
    return A








# G = nx.DiGraph()
# with open('C:\\Users\\Admin\\Downloads\\data\\p2p-Gnutella08.txt', 'r') as file:
#     for line in file:
#         if line.startswith("#"):
#             continue
#         u, v = map(int, line.strip().split('\t'))
#         G.add_edge(u, v)

# # Initialize random parameters for nodes
# num_topics = 3  # Number of topics (q)
# for node in G.nodes():
#     G.nodes[node]['c'] = 1
#     G.nodes[node]['state'] = set()
#     G.nodes[node]['p'] = {f'topic_{i+1}': random.uniform(0.1, 1) for i in range(num_topics)}
#     G.nodes[node]['gamma'] = {f'topic_{i+1}': random.uniform(0.1, 1) for i in range(num_topics)}
# for u, v in G.edges():
#     G[u][v]['weight'] = 1 / G.in_degree(v)
filepath = 'C:\\Users\\Admin\\Downloads\\data\\p2p-Gnutella08.txt'
num_topics = 3

my_graph = MyGraph(filepath=filepath, num_topics=3)

# Truy cập đồ thị G
G = my_graph.G

# Initialize the source sets
# Run the optimized IGA algorithm
budget = 20
T_reduced = 1  # Reduced number of Monte Carlo simulations
subgraph_ratio_reduced = 0.3  # Reduced ratio of nodes to keep in the subgraph
# A_final_subgraph_optimized = IGA_MT_LT_subgraph(G, source_sets, budget, T=T_reduced, subgraph_ratio=subgraph_ratio_reduced)

# print(sigma)

# Mô phỏng lan truyền thông tin cho từng chủ đề và lưu kết quả vào biến activated_graphs
activated_graphs = {}
# for topic in source_sets:
#     activated_graph , len_activateed= simulate_spread_MT_LT(G, source_sets, topic)
#     activated_graphs[topic] = len_activateed

# # In kết quả
# for topic, len_activateed in activated_graphs.items():
#     print(f'Activated nodes for topic {topic}: {len_activateed}')
# GEA_result_modified ,sigma = GEA_modified(G,list(activated_graph.nodes()), budget=20, q=3, T=3)

# print(GEA_result_modified)
# subgraph, num_nodes_by_topic = simulate_spread_MT_LT_updated_G(G, source_sets)

# Store the results
# all_results = {
#     'subgraph_edges': list(subgraph.edges()),
#     'num_activated_nodes_by_topic': num_nodes_by_topic
# }

# print(all_results)
# source_sets = {f'topic_{i+1}': [178 ] for i in range(num_topics)}

source_sets = {f'topic_{i+1}': random.sample(list(G.nodes()), min(100, len(G.nodes()))) for i in range(num_topics)}

print(source_sets)

updated_G, num_activated_nodes_by_topic = simulate_spread_MT_LT_updated_G(G.copy(), source_sets)

# Check the updated states for a subset of nodes
sample_nodes = random.sample(list(G.nodes()), min(10, len(G.nodes())))
print({node: G.nodes[node]['state'] for node in sample_nodes})
print(num_activated_nodes_by_topic)
# topics = ['topic_1', 'topic_2', 'topic_3']
# separated_graphs = separate_graphs_by_topic(updated_G, topics)

# # Find source nodes for each separated graph
# source_nodes_by_topic = {topic: [node for node in updated_G.nodes if topic in updated_G.nodes[node]['state']] for topic in topics}

# # Apply the unify_source_nodes function to each separated graph
# unified_graphs = {}
# for topic in topics:
#     unified_graphs[topic], Hi = unify_source_nodes_safe(separated_graphs[topic], source_nodes_by_topic[topic])

# Show the number of nodes and edges in the unified graphs for each topic
# unified_graphs_info = {topic: (len(G.nodes), len(G.edges)) for topic, G in unified_graphs.items()}
# unified_graphs_info
gea = GEA(updated_G, B = 20, num_samples=5)
print(gea)