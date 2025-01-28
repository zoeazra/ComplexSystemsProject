import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools

# initial parameters
N = 1000  # nodes
P = 0.0008  # collision probability
plow = 0.0001 # probability of generating new fragments
new_fragments_per_collision = 2  # debris per collision 
iterations = 1  # number of iterations, time steps

########################## A static without time network model
# genreate a probability list from 10^-4 to 10^-0.3
probabilities = np.logspace(-4, -0.3, num=500)
for p in probabilities:
    G2 = nx.random_graphs.erdos_renyi_graph(N, p)
    largest_cc2 = max(nx.connected_components(G2), key=len)
    avg_degree2 = np.mean([d for _, d in G2.degree()])
    gc_size2 = len(largest_cc2)
    print(f"Initial probability = {p}, GC2 ER network Size = {gc_size2}, Avg Degree = {avg_degree2:.2f}")

########################## A dynamic network model with time steps
# initialize the graph with N nodes, but no edges
G = nx.empty_graph(N)

def sample_collisions(nodes, P):
    """
    fast generation of collision pairs, avoiding explicit iteration over all pairs of nodes
    """
    N = len(nodes)
    num_possible_edges = N * (N - 1) // 2  # all the edges possible
    num_collisions = int(P * num_possible_edges)  # pair of nodes to collide

    # pick up the collision pairs randomly
    all_edges = list(itertools.combinations(nodes, 2))
    sampled_edges = np.random.choice(len(all_edges), size=num_collisions, replace=False)
    return [all_edges[i] for i in sampled_edges]

def direct_sample_collisions(nodes, P):
    """
    direct sampling of collision pairs, avoiding explicit iteration over all pairs of nodes
    """
    N = len(nodes)
    num_possible_edges = N * (N - 1) // 2
    num_collisions = int(P * num_possible_edges)

    sampled_indices = np.random.choice(num_possible_edges, size=num_collisions, replace=False)

    edges = []
    for idx in sampled_indices:
        i = int((-1 + np.sqrt(1 + 8 * idx)) // 2)  # inverse of the formula for the i-th triangular number
        j = idx - i * (i + 1) // 2
        edges.append((nodes[i], nodes[j]))
    return edges

for t in range(iterations):
    # generate collision pairs
    nodes = list(G.nodes)
    collision_edges = sample_collisions(nodes, P)
    G.add_edges_from(collision_edges)

    # generate new fragments
    for u, v in collision_edges:
        for _ in range(new_fragments_per_collision):
            if np.random.rand() < plow:
                new_node = len(G.nodes)
                G.add_node(new_node)
                print(f"New fragment {new_node} generated from collision between {u} and {v}, total nodes = {len(G.nodes)} \n")


    # GC size and average degree
    largest_cc = max(nx.connected_components(G), key=len)
    gc_size = len(largest_cc)
    avg_degree = np.mean([d for _, d in G.degree()])
    print(f"Iteration {t+1}: GC Size = {gc_size}, Avg Degree = {avg_degree:.2f}")
    
    #
    # plt.figure(figsize=(6, 6))
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, node_size=10, alpha=0.5)
    # nx.draw_networkx_nodes(G, pos, nodelist=largest_cc, node_color='red', node_size=20)
    # plt.title(f"Iteration {t+1}: GC Size = {gc_size}, Avg Degree = {avg_degree:.2f}")
    # plt.show()
