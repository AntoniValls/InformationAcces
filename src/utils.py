import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gzip
import scipy.sparse as sp
from tqdm import tqdm
import pandas as pd
import json

def plot_graph(G, title):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', edge_color='gray')
    plt.title(title)
    plt.show()
    
# Function to load the graph from a gzipped file
def load_graph_from_gzip(file_path):
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comment lines
            node1, node2 = line.strip().split()
            G.add_edge(node1, node2)
    return G

# Function to load a graph from a npz file
def load_dataset_from_npz(file_name):
    """Load a graph from a Numpy binary file and return a NetworkX graph.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    G : networkx.Graph
        A NetworkX graph with attributes and labels loaded from the file.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        # Create a NetworkX graph from the adjacency matrix
        G = nx.from_scipy_sparse_array(A)

        # Initialize all node attributes to zero
        num_nodes = A.shape[0]
        num_attrs = X.shape[1]
        for node in G.nodes():
            G.nodes[node].update({f'attr_{i}': 0.0 for i in range(num_attrs)})

        # Set the actual attributes from the sparse matrix
        X_coo = X.tocoo()
        for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
            G.nodes[i][f'attr_{j}'] = v

        # Add node labels
        if z is not None:
            labels = {i: z[i] for i in range(len(z))}
            nx.set_node_attributes(G, labels, 'label')

        # idx_to_node = loader.get('idx_to_node')
        # if idx_to_node is not None:
        #     idx_to_node = idx_to_node.tolist()
        #     node_id_mapping = {i: idx_to_node[i] for i in range(len(idx_to_node))}
        #     nx.relabel_nodes(G, node_id_mapping, copy=False)

        # idx_to_attr = loader.get('idx_to_attr')
        # if idx_to_attr is not None:
        #     idx_to_attr = idx_to_attr.tolist()
        #     for j, attr in enumerate(idx_to_attr):
        #         for i in range(X.shape[0]):
        #             if X[i, j] != 0:
        #                 G.nodes[i][attr] = X[i, j]

        # idx_to_class = loader.get('idx_to_class')
        # if idx_to_class is not None:
        #     idx_to_class = idx_to_class.tolist()
        #     class_labels = {i: idx_to_class[z[i]] for i in range(len(z))}
        #     nx.set_node_attributes(G, class_labels, 'class')

        return G

def load_dataset_from_csv(edges_file, features_file, target_file):
    """Load a graph from CSV and JSON files.

    Parameters
    ----------
    edges_file : str
        Path to the edges CSV file.
    features_file : str
        Path to the features JSON file.
    target_file : str
        Path to the target CSV file.

    Returns
    -------
    G : networkx.Graph
        A NetworkX graph with attributes and labels loaded from the files.
    """
    # Load edges
    edges_df = pd.read_csv(edges_file)
    edges = edges_df.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    G = nx.from_edgelist(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Load features
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    
    for node, features in features_data.items():
        G.nodes[int(node)]['features'] = features

    # Load targets
    target_df = pd.read_csv(target_file)
    target_dict = target_df.set_index('id')['target'].to_dict()
    nx.set_node_attributes(G, target_dict, 'label')

    return G

def get_total_classes(G):
    """Get the total number of unique classes in the graph.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph.

    Returns
    -------
    num_classes : int
        The total number of unique classes in the graph.
    """
    class_labels = nx.get_node_attributes(G, 'label')
    unique_classes = set(class_labels.values())
    num_classes = len(unique_classes)
    return num_classes

