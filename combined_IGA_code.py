
import networkx as nx
import random
from collections import defaultdict

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
                Gi_prime[Hi][v]['weight'] = Gi_prime[x][v]['weight'] * Gi_prime.nodes[x].get('p', {}).get('topic', 0)
            else:
                Gi_prime[Hi][v]['weight'] += Gi_prime[x][v]['weight'] * Gi_prime.nodes[x].get('p', {}).get('topic', 0)
    
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

def GEA_modified(G, S, budget, q, T=10):
    U = set(G.nodes())
    A1 = set()
    
    # Initialize sigma_tilde for each vertex
    sigma_tilde = defaultdict(int)
    
    # Create G_i's
    Gi_s = [G.copy() for _ in range(q)]
    Hi_s = []
    
    for i in range(q):
        Gi = Gi_s[i]
        Hi = f"Hi_{i}"
        Hi_s.append(Hi)
        Gi.add_node(Hi)
        for x in S:
            for v in Gi.successors(x):
                Gi.add_edge(Hi, v, weight=Gi[x][v]['weight'] * random.uniform(0, 1))
            
    
    # Monte Carlo simulation to generate T_i's
    Ti_s = []
    for i in range(q):
        Gi = Gi_s[i]
        Hi = Hi_s[i]
        Ti = []
        for _ in range(T):
            sample_tree = nx.DiGraph()
            sample_tree.add_node(Hi)
            to_visit = [(Hi, Hi)]
            while to_visit:
                parent, current = to_visit.pop()
                if current != parent:
                    sample_tree.add_edge(parent, current)
                for v in Gi.successors(current):
                    if random.random() < 1 / Gi.in_degree(v):
                        to_visit.append((current, v))
            Ti.append(sample_tree)
        Ti_s.append(Ti)
    
    # Calculate f(Ti, u) for all u in T
    f_values = {}
    for i in range(q):
        Ti = Ti_s[i]
        for tree in Ti:
            for u in tree.nodes():
                if u not in f_values:
                    f_values[u] = []
                f_values[u].append(f_dfs_iterative(tree, u))
    
    # Main loop
    while U:
        c_min = min(G.nodes[v]['c'] for v in U)
        if c_min + sum(G.nodes[v]['c'] for v in A1) > budget:
            break
        
        # Calculate delta for each vertex
        max_delta = -float('inf')
        u = None
        for v in U:
            delta_A1_v = 0
            if v in f_values:
                for f_val in f_values[v]:
                    delta_A1_v += f_val
            delta_A1_v /= (q * T)
            
            if delta_A1_v > max_delta:
                max_delta = delta_A1_v
                u = v
        
        if u is None:
            break

        U.remove(u)

        if sum(G.nodes[v]['c'] for v in A1) + G.nodes[u]['c'] <= budget:
            A1.add(u)
            for i in range(q):
                Ti = Ti_s[i]
                for tree in Ti:
                    if u in tree:
                        tree.remove_node(u)
                        for v in tree.nodes():
                            f_values[v] = [f_dfs_iterative(tree, v) for tree in Ti_s[i]]
                        
    # Calculate sigma_tilde for A1 and umax
    umax = max(f_values.keys(), key=lambda v: sum(f_values[v]) / len(f_values[v]) if v in f_values else 0)
    sigma_umax_tilde = sum(f_values[umax]) / len(f_values[umax]) if umax in f_values else 0
    sigma_A1_tilde = sum(sum(f_values[v]) / len(f_values[v]) for v in A1 if v in f_values)
    sigma_hat_A1 = 0
    for i in range(q):
        Ti = Ti_s[i]
        ni = len(Ti)
        sum_f_Ti_Ai = 0
        for tree in Ti:
            sum_f_Ti_Ai += f_dfs_iterative(tree, Hi_s[i])  # Assuming Hi_s[i] is the root for topic i
        sigma_hat_A1 += (sum_f_Ti_Ai / ni)
    sigma_hat_A1 /= q

    
    if sigma_umax_tilde > sigma_A1_tilde:
        return {umax} , sigma_umax_tilde
    else:
        return A1 , sigma_hat_A1







G = nx.DiGraph()
with open('C:\\Users\\Admin\\Downloads\\data\\p2p-Gnutella08.txt', 'r') as file:
    for line in file:
        if line.startswith("#"):
            continue
        u, v = map(int, line.strip().split('\t'))
        G.add_edge(u, v)

# Initialize random parameters for nodes
num_topics = 3  # Number of topics (q)
for node in G.nodes():
    G.nodes[node]['c'] = 1
    G.nodes[node]['state'] = set()
    G.nodes[node]['p'] = {f'topic_{i+1}': random.uniform(0.1, 1) for i in range(num_topics)}
    G.nodes[node]['gamma'] = {f'topic_{i+1}': random.uniform(0.1, 1) for i in range(num_topics)}
for u, v in G.edges():
    G[u][v]['weight'] = 1 / G.in_degree(v)

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

random_set = random.sample(list(G.nodes()), min(100, len(G.nodes())))
source_sets = {f'topic_{i+1}': random_set for i in range(num_topics)}

print(source_sets)

updated_G, num_activated_nodes_by_topic = simulate_spread_MT_LT_updated_G(G.copy(), source_sets)

# Check the updated states for a subset of nodes
# sample_nodes = random.sample(list(G.nodes()), min(10, len(G.nodes())))
# print({node: G.nodes[node]['state'] for node in sample_nodes})
# print(num_activated_nodes_by_topic)
topics = ['topic_1', 'topic_2', 'topic_3']
separated_graphs = separate_graphs_by_topic(updated_G, topics)

# Find source nodes for each separated graph
source_nodes_by_topic = {topic: [node for node in updated_G.nodes if topic in updated_G.nodes[node]['state']] for topic in topics}

# Apply the unify_source_nodes function to each separated graph
unified_graphs = {}
for topic in topics:
    unified_graphs[topic], Hi = unify_source_nodes_safe(separated_graphs[topic], source_nodes_by_topic[topic])

# Show the number of nodes and edges in the unified graphs for each topic
unified_graphs_info = {topic: (len(G.nodes), len(G.edges)) for topic, G in unified_graphs.items()}
unified_graphs_info
