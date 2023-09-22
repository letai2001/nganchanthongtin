import networkx as nx

def unify_source_nodes(Gi, Si):
    # Step 1: Create a copy of the original graph
    Gi_prime = Gi.copy()
    
    # Step 2: Add a new node Hi to Gi_prime
    Hi = 'Hi'
    Gi_prime.add_node(Hi)
    
    # Step 3: Update edges and weights
    for x in Si:
        for v in list(Gi_prime.successors(x)):
            if (Hi, v) not in Gi_prime.edges():
                # Step 6: Add edge (Hi, v) to Gi_prime
                Gi_prime.add_edge(Hi, v)
                
                # Step 7: Update weight wi_prime(Hi, v) = wi(x, v) * pxi
                Gi_prime[Hi][v]['weight'] = Gi_prime[x][v]['weight'] * Gi_prime.nodes[x].get('p', {}).get('topic', 0)
            else:
                # Step 9: Update weight wi_prime(Hi, v)
                Gi_prime[Hi][v]['weight'] += Gi_prime[x][v]['weight'] * Gi_prime.nodes[x].get('p', {}).get('topic', 0)
    
    # Step 14: Remove all nodes Si from Gi_prime
    Gi_prime.remove_nodes_from(Si)
    
    return Gi_prime, Hi
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


# Test with a sample graph and source set
G_sample = nx.DiGraph()
G_sample.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
for u, v in G_sample.edges():
    G_sample[u][v]['weight'] = 1.0
G_sample.nodes[1]['p'] = {'topic': 0.6}
G_sample.nodes[2]['p'] = {'topic': 0.7}
G_sample.nodes[3]['p'] = {'topic': 0.8}

S_sample = [1, 2, 3]

# Run the function
G_prime_sample, Hi_sample = unify_source_nodes_safe(G_sample, S_sample)

# Output the new graph and new source node
print(G_prime_sample.edges(data=True), Hi_sample)
