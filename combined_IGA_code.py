
import networkx as nx
import random
from collections import defaultdict

# Function to simulate the spread of misinformation based on the MT-LT model
def simulate_spread_MT_LT(G, sources, topic):
    activated_nodes = set()
    for source in sources.get(topic, []):
        if source in G.nodes:
            activated_nodes.add((source, topic))
    
    # Khởi tạo một đồ thị con
    subgraph = nx.DiGraph()
    
    while True:
        new_activations = set()
        for source, current_topic in list(activated_nodes):
            if source not in G.nodes:
                continue
            for neighbor in G.successors(source):
                if (neighbor, current_topic) not in activated_nodes and neighbor in G.nodes:
                    influence_sum = sum(G[source][neighbor]['weight'] * G.nodes[u]['p'].get(t, 0) for u, t in activated_nodes if t == current_topic)
                    # print(G.nodes[u]['p'].get(current_topic, 0))

                    # print(influence_sum)
                    if influence_sum >= G.nodes[neighbor]['gamma'].get(current_topic, 0):
                        # print(G.nodes[neighbor]['gamma'].get(current_topic, 0))
                        new_activations.add((neighbor, current_topic))
                        subgraph.add_edge(source, neighbor)  # Thêm cạnh vào đồ thị con
        if not new_activations:
            break
        activated_nodes.update(new_activations)
    
    return subgraph , len(activated_nodes)
def simulate_spread(G, sources):
    activated_nodes = set(sources)
    for source in sources:
        for neighbor in G.successors(source):
            if neighbor not in activated_nodes:
                if random.random() < 1 / G.in_degree(neighbor):
                    activated_nodes.add(neighbor)
    return len(activated_nodes)


def monte_carlo_simulation(G, sources, T=10):
    count = 0
    for _ in range(T):
        Ni = simulate_spread(G, sources)
        count += Ni
    return count / T


# Monte Carlo simulation with subgraph sampling
def monte_carlo_simulation_MT_LT_subgraph(G, source_sets, T=10, subgraph_ratio=0.7):
    count = 0
    for _ in range(T):
        sampled_nodes = random.sample(G.nodes(), int(len(G.nodes()) * subgraph_ratio))
        subG = G.subgraph(sampled_nodes)
        sub_source_sets = {topic: [s for s in sources if s in subG.nodes] for topic, sources in source_sets.items()}
        
        for topic, sources in sub_source_sets.items():
            Ni = simulate_spread_MT_LT(subG, sub_source_sets, topic)
            count += Ni
    return count / (T * len(source_sets))

# IGA Algorithm with MT-LT model and subgraph sampling
def IGA_MT_LT_subgraph(G, source_sets, budget, T=10, subgraph_ratio=0.7):
    A1 = set()
    U = set(G.nodes())
    initial_D = monte_carlo_simulation_MT_LT_subgraph(G, source_sets, T=T, subgraph_ratio=subgraph_ratio)

    vmax = None
    max_sigma_vmax = -float('inf')

    for v in G.nodes():
        if G.nodes[v]['c'] <= budget:
            D_v = monte_carlo_simulation_MT_LT_subgraph(G, {topic: [v] for topic in source_sets.keys()}, T=T, subgraph_ratio=subgraph_ratio)
            sigma_v = initial_D - D_v
            if sigma_v > max_sigma_vmax:
                max_sigma_vmax = sigma_v
                vmax = v

    if vmax is not None:
        A1.add(vmax)

    while U:
        u = None
        max_delta = -float('inf')
        for v in U - A1:
            D_A1_u = monte_carlo_simulation_MT_LT_subgraph(G, {topic: list(A1) + [v] for topic in source_sets.keys()}, T=T, subgraph_ratio=subgraph_ratio)
            sigma_A1_u = initial_D - D_A1_u
            delta_u = (sigma_A1_u - max_sigma_vmax) / G.nodes[v]['c']
            if delta_u > max_delta:
                max_delta = delta_u
                u = v

        if u is not None and (sum(G.nodes[v]['c'] for v in A1) + G.nodes[u]['c']) <= budget:
            A1.add(u)

        if u is not None:
            U.remove(u)

    D_A1 = monte_carlo_simulation_MT_LT_subgraph(G, {topic: list(A1) for topic in source_sets.keys()}, T=T, subgraph_ratio=subgraph_ratio)
    sigma_A1 = initial_D - D_A1

    if sigma_A1 >= max_sigma_vmax:
        return A1
    else:
        return {vmax}

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
    G.nodes[node]['state'] = 'inactive'
    G.nodes[node]['p'] = {f'topic_{i+1}': random.uniform(0, 1) for i in range(num_topics)}
    G.nodes[node]['gamma'] = {f'topic_{i+1}': random.uniform(0, 1) for i in range(num_topics)}

# Initialize the source sets
source_sets = {f'topic_{i+1}': random.sample(list(G.nodes), 3) for i in range(num_topics)}
print(source_sets)
# Run the optimized IGA algorithm
budget = 20
T_reduced = 1  # Reduced number of Monte Carlo simulations
subgraph_ratio_reduced = 0.3  # Reduced ratio of nodes to keep in the subgraph
# A_final_subgraph_optimized = IGA_MT_LT_subgraph(G, source_sets, budget, T=T_reduced, subgraph_ratio=subgraph_ratio_reduced)
for u, v in G.edges():
    G[u][v]['weight'] = 1 / G.in_degree(v)

# print(sigma)

# Mô phỏng lan truyền thông tin cho từng chủ đề và lưu kết quả vào biến activated_graphs
activated_graphs = {}
for topic in source_sets:
    activated_graph , len_activateed= simulate_spread_MT_LT(G, source_sets, topic)
    activated_graphs[topic] = activated_graph

# In kết quả
for topic, len_activateed in activated_graphs.items():
    print(f'Activated nodes for topic {topic}: {len_activateed}')
GEA_result_modified ,sigma = GEA_modified(G,list(activated_graph.nodes()), budget=20, q=3, T=3)

print(GEA_result_modified)
