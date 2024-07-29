# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Diffusion Measures from Jackson (2020) paper
# In this notebook we try to reproduce the diffusion measures presented in the Jackson et. al. paper (2020). In this paper Jackson breaks down the term _"social capital"_ into seven more fundamental forms of capital: information capital, brokerage capital, coordination and leadership capital, bridging capital, favor capital, reputation capital, and community capital.
#
# For them, it defines different diffusion measures:
#
# 1. **Decay centrality**: It counts paths of different lengths—so how many people one can reach at different distances—and weights them by their distance.
#
# $$ \text{Dec}_i(\textbf{g},p,T) = \sum_{l=1}^T p^l|N_i^l(g)| $$
#
# Here, the parameter $p$ captures the decay of information with distance, $T$ caps the number of time that information is relayes, and $|N_i^l(g)|$ denotes the number of nodes at distance $l$ from node $i$.
#
# An advantage of decay centrality is that it is easy to measure. A disadvantage of decay centrality is that it does not account for the fact that in some settings it is easier to reach any other given person, either in sending or receiving information, if there are many independent paths to that person. Multiple paths can increase the chance that the information makes it from one node to another. This was the motivation behind the concept of communication centrality:
#
# 2. **Communication centrality**: Define $\text{PInf}(\textbf{p}, T )_{ij}$ to be the probability that node $j$ ends up hearing information that starts from node $i$ if it is passed independently with probability $p_{i′ j′}$ from node $i′$ to $j′$ along each walk in the network, and running the whole process for $T$ periods. Then:
#
#
# $$ \text{Com}_i(\textbf{p},T) = \sum_{j}\text{PInf}(\textbf{p},T)_{ij} $$
#
# Here **p** is a $n × n$ weighted and directed network with $ij$-th entry being the probability that $i$ passes information to $j$ in a given time period. A special case is in which $p_{ij}$’s are identical for all $ij$ and then $p$ becomes p$g$, where p is a scalar and $g$ is an adjacency matrix taking on values {0, 1}. 
#
# 3. **Diffusuon centrality**
#
# $$ \text{Diff}_i(\textbf{p},T) = \sum_{j}\text{EInf}(\textbf{p},T)_{ij} = \sum_j\sum_{l=1}^T[\textbf{p}^l]_{ij} $$
#
#

# +
import networkx as nx
import matplotlib.pyplot as plt
from src.utils import plot_graph

G = nx.barabasi_albert_graph(50, 2)

plot_graph(G, "Graph")
# Print some basic information about the graph
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Graph has {num_nodes} nodes and {num_edges} edges.")


# +
from src.jackson_metrics import decay_centrality, communication_centrality, diffusion_centrality

p = 0.5
T = 2

decay_centrality = decay_centrality(G, p, T)
communication_centrality = communication_centrality(G, T)
diffusion_centrality = diffusion_centrality(G, T)

# +
import cmocean

print(f"T = {T}, p = {p}\n")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  

cmap = cmocean.cm.thermal
# Graph Colored by Decay Centrality
pos = nx.spring_layout(G, seed=20)
node_color = [decay_centrality[node] for node in G.nodes]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=cmap, node_size=500, ax=axs[0])
edges = nx.draw_networkx_edges(G, pos, edge_color='gray', ax=axs[0])
labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=axs[0])
fig.colorbar(nodes, ax=axs[0])
axs[0].set_title("Graph Colored by Decay Centrality")
axs[0].axis('off')

# Graph Colored by Communication Centrality
node_color = [communication_centrality[node] for node in G.nodes]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=cmap, node_size=500, ax=axs[1])
edges = nx.draw_networkx_edges(G, pos, edge_color='gray', ax=axs[1])
labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=axs[1])
fig.colorbar(nodes, ax=axs[1])
axs[1].set_title("Graph Colored by Communication Centrality")
axs[1].axis('off')

# Graph Colored by Diffusion Centrality
node_color = [diffusion_centrality[node] for node in G.nodes]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=cmap, node_size=500, ax=axs[2])
edges = nx.draw_networkx_edges(G, pos, edge_color='gray', ax=axs[2])
labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=axs[2])
fig.colorbar(nodes, ax=axs[2])
axs[2].set_title("Graph Colored by Diffusion Centrality")
axs[2].axis('off')

plt.tight_layout()
plt.show()

# -


