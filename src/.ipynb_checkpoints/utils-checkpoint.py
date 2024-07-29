# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gzip

# Function to load the graph from a gzipped file
def load_graph(file_path):
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comment lines
            node1, node2 = line.strip().split()
            G.add_edge(node1, node2)
    return G

def plot_graph(G, title):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', edge_color='gray')
    plt.title(title)
    plt.show()


def generate_twitter_like_graph(num_nodes, avg_out_degree):
    
    # Generate an undirected Barabási-Albert graph
    m = avg_out_degree // 2  # Number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(num_nodes, m)

    # Convert to directed graph by assigning direction to each edge randomly
    DG = nx.DiGraph()
    for u, v in G.edges():
        if np.random.rand() > 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)

    return DG
