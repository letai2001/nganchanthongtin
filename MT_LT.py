import networkx as nx
import random


def simulate_spread_MT_LT(G, sources, topic):
    activated_nodes = set()
    for source in sources.get(topic, []):
        if source in G.nodes:
            activated_nodes.add((source, topic))
    
    # Simulate the spread for the specific topic
    for source, current_topic in list(activated_nodes):  # Convert to list to prevent "Set changed size during iteration" error
        if source not in G.nodes:
            continue
        for neighbor in G.successors(source):
            if (neighbor, current_topic) not in activated_nodes and neighbor in G.nodes:
                influence_sum = sum(G[source][neighbor].get('w', {}).get(t, 0) for s, t in activated_nodes if t == current_topic)
                if influence_sum >= G.nodes[neighbor]['gamma'].get(current_topic, 0):
                    activated_nodes.add((neighbor, current_topic))
    
    return len(activated_nodes)


# def monte_carlo_simulation(G, sources, T=10):
#     count = 0
#     for _ in range(T):
#         Ni = simulate_spread_MT_LT(G, sources)
#         count += Ni
#     return count / T
def monte_carlo_simulation_MT_LT_subgraph(G, source_sets, T=10, subgraph_ratio=0.7):
    count = 0
    for _ in range(T):
        # Create a random subgraph
        sampled_nodes = random.sample(G.nodes(), int(len(G.nodes()) * subgraph_ratio))
        subG = G.subgraph(sampled_nodes)
        
        # Update source_sets for the subgraph
        sub_source_sets = {topic: [s for s in sources if s in subG.nodes] for topic, sources in source_sets.items()}
        
        for topic, sources in sub_source_sets.items():
            Ni = simulate_spread_MT_LT(subG, sub_source_sets, topic)
            count += Ni
    return count / (T * len(source_sets))


# Initialize the graph
G = nx.DiGraph()
with open('C:\\Users\\Admin\\Downloads\\data\\p2p-Gnutella08.txt', 'r') as file:
    for line in file:
        if line.startswith("#"):
            continue
        u, v = map(int, line.strip().split('\t'))
        G.add_edge(u, v)

# Initialize random parameters for nodes
for node in G.nodes():
    G.nodes[node]['c'] = random.uniform(1.0, 3.0)
    G.nodes[node]['state'] = 'inactive'
    G.nodes[node]['p'] = {f'topic_{i+1}': random.uniform(0, 1) for i in range(3)}
    G.nodes[node]['gamma'] = {f'topic_{i+1}': random.uniform(0, 1) for i in range(3)}

# Initialize the source sets S1, S2, S3
source_sets = {f'S{i+1}': random.sample(list(G.nodes), 100) for i in range(3)}

# Set the budget
budget = 20

# IGA Algorithm
A1 = set()
U = set(G.nodes())
initial_D = sum(monte_carlo_simulation_MT_LT_subgraph(G, sources, T=2) for sources in source_sets.values())

# vmax = None
# max_sigma_vmax = -float('inf')

# for v in G.nodes():
#     if G.nodes[v]['c'] <= budget:
#         D_v = sum(monte_carlo_simulation_MT_LT_subgraph(G, [v], T=2) for _ in source_sets.values())
#         sigma_v = initial_D - D_v
#         if sigma_v > max_sigma_vmax:
#             max_sigma_vmax = sigma_v
#             vmax = v

# if vmax is not None:
#     A1.add(vmax)

# while U:
#     u = None
#     max_delta = -float('inf')
    
#     for v in U - A1:
#         D_A1_u = sum(monte_carlo_simulation(G, list(A1) + [v], T=2) for _ in source_sets.values())
#         sigma_A1_u = initial_D - D_A1_u
#         delta_u = (sigma_A1_u - max_sigma_vmax) / G.nodes[v]['c']
        
#         if delta_u > max_delta:
#             max_delta = delta_u
#             u = v

#     if u is not None and (sum(G.nodes[v]['c'] for v in A1) + G.nodes[u]['c']) <= budget:
#         A1.add(u)

#     if u is not None:
#         U.remove(u)

# D_A1 = sum(monte_carlo_simulation(G, list(A1), T=2) for _ in source_sets.values())
# sigma_A1 = initial_D - D_A1

# if sigma_A1 >= max_sigma_vmax:
#     A_final = A1
# else:
#     A_final = {vmax}

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

# Test IGA algorithm with MT-LT model and subgraphs
budget = 20
T = 2  # Number of Monte Carlo simulations
subgraph_ratio = 0.5  # Ratio of nodes to keep in the subgraph
A_final_subgraph = IGA_MT_LT_subgraph(G, source_sets, budget, T=T, subgraph_ratio=subgraph_ratio)