# -*- coding: utf-8 -*-
"""Compare algorithms.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KqIrqj82vJXAeI_2anPh-L4yBwzjCIjU

# Louvain Algorithm
"""

!pip install louvain

import louvain
import igraph as ig
import time
import networkx as nx

# Calculate modularity and runtime using Louvain Algorithm
def analyze_graph_louvain(dataset_path):
    start_time = time.time()
    G = ig.Graph.Read_Ncol(dataset_path, directed=True)
    part = louvain.find_partition(G, louvain.ModularityVertexPartition)
    end_time = time.time()

    print("Is the graph connected?", G.is_connected())
    print('Modularity_louvain:', part.modularity)
    print("Runtime_louvain:", end_time - start_time, "seconds")
    print("Number of clusters:", len(part))
    print("Number of elements:", G.vcount())
    return G, part.membership

# Calculate the maximal modularity after 5 replications of 10 iterations each
def compute_modularity_louvain(G):
    start_time = time.time()

    optimiser = louvain.Optimiser()

    max_modularity = -1
    best_partition_louvain = None

    for _ in range(5):  # 5 replications
        part = louvain.ModularityVertexPartition(G)
        for _ in range(10):  # 10 iterations each
            improvement = optimiser.optimise_partition(part)
            if not improvement:
                break  # stop if no improvement
        if part.modularity > max_modularity:
            max_modularity = part.modularity
            best_partition_louvain = part

    end_time = time.time()

    print('Max Modularity_louvain:', max_modularity)
    print("Runtime_louvain:", end_time - start_time, "seconds")
    return best_partition_louvain.membership

"""# Leiden Algorithm"""

!pip install leidenalg

import leidenalg
import igraph as ig
import time

# Calculate modularity and runtime using Leiden Algorithm
def analyze_graph_leiden(dataset_path):
    start_time = time.time()
    G = ig.Graph.Read_Ncol(dataset_path, directed=True)
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    end_time = time.time()

    print("Is the graph connected?", G.is_connected())
    print('Modularity_leiden:', part.modularity)
    print("Runtime_leiden:", end_time - start_time, "seconds")
    print("Number of clusters:", len(part))
    print("Number of elements:", G.vcount())
    return G, part.membership

# Calculate the maximal modularity after 5 replications of 10 iterations each
def compute_modularity_leiden(G):
    start_time = time.time()

    optimiser = leidenalg.Optimiser()

    max_modularity = -1
    best_partition_leiden = None

    for _ in range(5):  # 5 replications
        part = leidenalg.ModularityVertexPartition(G)
        for _ in range(10):  # 10 iterations each
            improvement = optimiser.optimise_partition(part, n_iterations=1)
            if not improvement:
                break  # stop if no improvement
        if part.modularity > max_modularity:
            max_modularity = part.modularity
            best_partition_leiden = part

    end_time = time.time()

    print('Max Modularity_leiden:', max_modularity)
    print("Runtime_leiden:", end_time - start_time, "seconds")
    return best_partition_leiden.membership

"""# Experiements on networks"""

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

"""## Flickr"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /soc-flickr.mtx'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /soc-flickr_clean.mtx'

with open(dataset_path, 'r') as f_in, open(clean_dataset_path, 'w') as f_out:
    next(f_in)  # skip first line
    next(f_in)  # skip second line
    for line in f_in:
        f_out.write(line)

G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_flickr = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_flickr}")

ari_flickr = adjusted_rand_score(labels_leiden, labels_louvain)

print(f"Adjusted Rand Index: {ari_flickr}")

"""## DBLP"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /com-dblp.ungraph.txt'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /com-dblp.ungraph_clean.txt'

with open(dataset_path, 'r') as f_in:
    lines = f_in.readlines()

with open(clean_dataset_path, 'w') as f_out:
    for line in lines:
        if not line.startswith('#'):  # skip comment lines
            f_out.write(line)


G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_dblp = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_dblp}")

ari_dblp = adjusted_rand_score(labels_leiden, labels_louvain)

print(f"Adjusted Rand Index: {ari_dblp}")

"""## Youtube"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /com-youtube.ungraph.txt'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /com-youtube.ungraph_clean.txt'

with open(dataset_path, 'r') as f_in:
    lines = f_in.readlines()

with open(clean_dataset_path, 'w') as f_out:
    for line in lines:
        if not line.startswith('#'):  # skip comment lines
            f_out.write(line)


G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_youtube = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_youtube}")

ari_youtube = adjusted_rand_score(labels_leiden, labels_louvain)

print(f"Adjusted Rand Index: {ari_youtube}")

"""## Amazon"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /com-amazon.ungraph.txt'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /com-amazon.ungraph_clean.txt'

with open(dataset_path, 'r') as f_in:
    lines = f_in.readlines()

with open(clean_dataset_path, 'w') as f_out:
    for line in lines:
        if not line.startswith('#'):  # skip comment lines
            f_out.write(line)


G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_amazon = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_amazon}")

ari_amazon = adjusted_rand_score(labels_leiden, labels_louvain)

print(f"Adjusted Rand Index: {ari_amazon}")

"""## Gnutella peer-to-peer network"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /p2p-Gnutella04.txt'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /p2p-Gnutella04_clean.txt'

with open(dataset_path, 'r') as f_in:
    lines = f_in.readlines()

with open(clean_dataset_path, 'w') as f_out:
    for line in lines:
        if not line.startswith('#'):  # skip comment lines
            f_out.write(line)


G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_gnutella = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_gnutella}")

ari_gnutella = adjusted_rand_score(labels_leiden, labels_louvain)

print(f"Adjusted Rand Index: {ari_gnutella}")

"""## Epinions social network"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /soc-Epinions1.txt'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /soc-Epinions1_clean.txt'
with open(dataset_path, 'r') as f_in:
    lines = f_in.readlines()

with open(clean_dataset_path, 'w') as f_out:
    for line in lines:
        if not line.startswith('#'):  # skip comment lines
            f_out.write(line)

G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_epinions = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_epinions}")

ari_epinions = adjusted_rand_score(labels_leiden, labels_louvain)

print(f"Adjusted Rand Index: {ari_epinions}")

"""## ca-MathSciNet"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /ca-MathSciNet.mtx'

# Open the file in read mode
with open(dataset_path, 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

G, labels_louvain = analyze_graph_louvain(dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_ca_math = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_ca_math}")

ari_ca_math = adjusted_rand_score(labels_leiden, labels_louvain)
print(f"Adjusted Rand Index: {ari_ca_math}")

"""## Twitter"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /twitter_combined.txt'

# Open the file in read mode
with open(dataset_path, 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

G, labels_louvain = analyze_graph_louvain(dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_twitter = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_twitter}")

ari_twitter = adjusted_rand_score(labels_leiden, labels_louvain)
print(f"Adjusted Rand Index: {ari_twitter}")

"""## Slashdot"""

dataset_path = '/content/drive/MyDrive/SNACS/Paper /soc-Slashdot0811.txt'
clean_dataset_path = '/content/drive/MyDrive/SNACS/Paper /soc-Slashdot0811_clean.txt'
with open(dataset_path, 'r') as f_in:
    lines = f_in.readlines()

with open(clean_dataset_path, 'w') as f_out:
    for line in lines:
        if not line.startswith('#'):  # skip comment lines
            f_out.write(line)

G, labels_louvain = analyze_graph_louvain(clean_dataset_path)
best_partition_louvain = compute_modularity_louvain(G)

print('********************')

G, labels_leiden = analyze_graph_leiden(clean_dataset_path)
best_partition_leiden = compute_modularity_leiden(G)

labels_leiden = best_partition_leiden
labels_louvain = best_partition_louvain

nmi_slashdot = normalized_mutual_info_score(labels_leiden, labels_louvain)
print(f"Normalized Mutual Information: {nmi_slashdot}")

ari_slashdot = adjusted_rand_score(labels_leiden, labels_louvain)
print(f"Adjusted Rand Index: {ari_slashdot}")

"""# Plot ARI and NMI values"""

# Store the NMI values
nmi_values = {
    'Flickr': nmi_flickr,
    'DBLP': nmi_dblp,
    'Youtube': nmi_youtube,
    'Amazon': nmi_amazon,
    'Gnutella': nmi_gnutella,
    'Epinions': nmi_epinions,
    'ca-Math': nmi_ca_math,
    'twitter': nmi_twitter,
    'slashdot': nmi_slashdot
}

# Store the ARI values
ari_values = {
    'Flickr': ari_flickr,
    'DBLP': ari_dblp,
    'Youtube': ari_youtube,
    'Amazon': ari_amazon,
    'Gnutella': ari_gnutella,
    'Epinions': ari_epinions,
    'ca-Math': ari_ca_math,
    'twitter': ari_twitter,
    'slashdot': ari_slashdot
}

# Store NMI value in a list
nmi_values = [nmi_flickr, nmi_dblp, nmi_youtube, nmi_amazon, nmi_gnutella, nmi_epinions, nmi_ca_math, nmi_twitter, nmi_slashdot]

# Store ARI value in a list
ari_values = [ari_flickr, ari_dblp, ari_youtube, ari_amazon, ari_gnutella, ari_epinions, ari_ca_math, ari_twitter, ari_slashdot]

networks = ['Flickr', 'DBLP', 'Youtube', 'Amazon', 'Gnutella', 'Epinions', 'ca_Math', 'twitter', 'slashdot']

# Create a scatter plot
plt.figure(figsize=(9, 5))

# Plot ARI values
plt.scatter(networks, ari_values, color='red', label='ARI')

# Plot NMI values
plt.scatter(networks, nmi_values, color='blue', label='NMI')

plt.xlabel('Network')
plt.ylabel('Value')
plt.title('ARI and NMI Values for Different Networks')
plt.legend()
plt.show()