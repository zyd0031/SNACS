#!/data/s3817598/envs/social/bin/python


import igraph as ig
import pandas as pd
import time
import os

data_path = "/data/s3817598/snacs/data"
output_path = "/data/s3817598/snacs/results"

def create_graph(file_name):
	file_path = os.path.join(data_path, file_name)
	graph = ig.Graph.Read_Ncol(file_path, directed = False)
	return graph

def basic_statistics(graph, graph_name):
	nodes = graph.vcount()
	edges = graph.ecount()
	density = graph.density()
	print(f"{graph_name} has {nodes} nodes, {edges} edges and the density is {density}")
	

def louvain_algorithm(graph, graph_name):

	start_time = time.time()
	communities = graph.community_multilevel()
	end_time = time.time()
	run_time = end_time - start_time
	print(f"{graph_name} Louvain Algorithm Run Time:{run_time}")
	return communities


def evaluate_community_structure(graph, communities):
	results = []

	for community in communities:
		subgraph = graph.subgraph(community)
		nodes = subgraph.vcount()
		edges = subgraph.ecount()
		density = subgraph.density()
		average_degree = sum(subgraph.degree()) / len(community)
		# average_clustering_coefficient = sum(subgraph.transitivity_local_undirected(vertices = None)) / len(community)
		isolated_nodes = len([node for node in range(len(community)) if subgraph.degree(node) == 0])
		proportion_isolated = isolated_nodes / len(community)

		results.append({
			"Nodes": nodes,
			"Edges": edges,
			"Density": density,
			"Average Degree": average_degree,
			# "Average Clustering Coefficient": average_clustering_coefficient,
			"Isolated Nodes": isolated_nodes,
			"Proportion of Isolated Nodes": proportion_isolated
		})

	df = pd.DataFrame(results)
	modularity = communities.modularity
	df["Modularity"] = modularity
	return df

def run_louvain(graph, graph_name, output_name):
	communities = louvain_algorithm(graph, graph_name)
	df = evaluate_community_structure(graph, communities)
	path = os.path.join(output_path, output_name)
	df.to_csv(path)
def load_ground_truth(file_name):
	file_path = os.path.join(data_path, file_name)
	with open(file_path, 'r') as file:
		ground_truth = [list(map(int, line.split())) for line in file]
	return ground_truth

def calculate_isolated_rates(df):
	percentage = (df[df["Isolated Nodes"] != 0].shape[0] / df.shape[0]) * 100
	print(f"{percentage:.2f}% communities have isolated nodes.")

def get_graph_and_community(path):
	graph = create_graph(path)
	community = graph.community_multilevel()
	return graph, community
			
