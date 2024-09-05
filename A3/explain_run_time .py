#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils
import pandas as pd
import igraph as ig
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


math_sci_name = "ca-MathSciNet.mtx"
math_sci, math_sci_louvain_communities= utils.get_graph_and_community(math_sci_name)


# In[3]:


utils.basic_statistics(math_sci, "math_sci")


# In[11]:


math_sci_average_clustering_coefficient = math_sci.transitivity_undirected()
math_sci_average_clustering_coefficient


# In[3]:


amazon_name = "com-amazon.ungraph.txt"
amazon, amazon_louvain_communities= utils.get_graph_and_community(amazon_name)


# In[5]:


utils.basic_statistics(amazon, "amazon")


# In[4]:


youtube_name = "com-youtube.ungraph.txt"
youtube, youtube_louvain_communities= utils.get_graph_and_community(youtube_name)


# In[7]:


utils.basic_statistics(youtube, "youtube")


# In[5]:


dblp_name = "com-dblp.ungraph.txt"
dblp, dblp_louvain_communities= utils.get_graph_and_community(dblp_name)


# In[9]:


utils.basic_statistics(dblp, "dblp")


# In[6]:


flickr_name = "soc-flickr.mtx"
flickr, flickr_louvain_communities= utils.get_graph_and_community(flickr_name)


# In[16]:


utils.basic_statistics(flickr, "flickr")


# In[12]:


flickr_average_clustering_coefficient = flickr.transitivity_undirected()
flickr_average_clustering_coefficient


# In[19]:


dblp_degree = dblp.degree()
math_sci_degree = math_sci.degree()
amazon_degree = amazon.degree()
flickr_degree = flickr.degree()
youtube_degree = youtube.degree()


# In[21]:


for i in [dblp_degree, math_sci_degree, amazon_degree, flickr_degree, youtube_degree]:
    print(np.mean(i))


# In[24]:


for i in [dblp_degree, math_sci_degree, amazon_degree, flickr_degree, youtube_degree]:
    print(np.max(i))


# In[37]:


fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

bins = np.linspace(0, 600, 50)
axs[0].hist(dblp_degree, bins=bins, alpha=0.3, label='dblp', log=True)
axs[0].hist(math_sci_degree, bins=bins, alpha=0.3, label='math_sci', log=True)
axs[0].hist(amazon_degree, bins=bins, alpha=0.3, label='amazon', log=True)
axs[0].legend(loc='upper right')

bins = np.linspace(0, 4369, 100)
axs[1].hist(flickr_degree, bins=bins, alpha=0.3, label='flickr', log=True)
axs[1].legend(loc='upper right')

bins = np.linspace(0, 28754, 100)
axs[2].hist(youtube_degree, bins=bins, alpha=0.3, label='youtube', log=True)
axs[2].legend(loc='upper right')


fig.text(0.5, -0.01, 'Degree', ha='center', va='center')
fig.text(-0.01, 0.5, 'Frequency (log scale)', ha='center', va='center', rotation='vertical')

plt.tight_layout()

plt.show()


# In[7]:


dblp_df = utils.evaluate_community_structure(dblp, dblp_louvain_communities)


# In[8]:


math_sci_df = utils.evaluate_community_structure(math_sci, math_sci_louvain_communities)


# In[9]:


amazon_df = utils.evaluate_community_structure(amazon, amazon_louvain_communities)


# In[10]:


flickr_df = utils.evaluate_community_structure(flickr, flickr_louvain_communities)


# In[11]:


youtube_df = utils.evaluate_community_structure(youtube, youtube_louvain_communities)


# In[21]:


import os

path_ = "/data/s3817598/snacs/results"
def save_to_csv(df, name):
    file_path = os.path.join(path_, name + ".csv")
    df.to_csv(file_path, index = False)


# In[22]:


for df, name in zip([dblp_df, math_sci_df, amazon_df, flickr_df, youtube_df], ["dnlp", "math_sci", "amazon", "flickr", "youtube"]):
    save_to_csv(df, name)


# In[22]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

def plot_with_fit(df, ax, title):
    z_scores = zscore(df[["Nodes", "Edges"]])
    df_no_outliers = df[(np.abs(z_scores) < 3).all(axis=1)]  # Keep data points within 3 standard deviations
    
    sns.scatterplot(data=df, x="Nodes", y="Edges", ax=ax)
    
    ax.set_title(title)
    
    m, b = np.polyfit(df_no_outliers["Nodes"], df_no_outliers["Edges"], 1)
    
    x_range = np.linspace(df["Nodes"].min(), df["Nodes"].max() + 10, 100)
    
    ax.plot(x_range, m*x_range + b, color='red', linestyle='--')  # Use the extended x-range
    ax.text(0.35, 0.95, f'Slope: {m:.2f}', transform=ax.transAxes, color='red')

    ax.legend()

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

plot_with_fit(dblp_df, axes[0], "dblp")
plot_with_fit(math_sci_df, axes[1], "math_sci")
plot_with_fit(amazon_df, axes[2], "amazon")
plot_with_fit(flickr_df, axes[3], "flickr")
plot_with_fit(youtube_df, axes[4], "youtube")

plt.tight_layout()
plt.show()


# In[55]:


df_all = pd.concat([df['Density'].rename(df_name) for df, df_name in zip([dblp_df, math_sci_df, amazon_df, flickr_df, youtube_df], ['dblp', 'math_sci', 'amazon', 'flickr', 'youtube'])], axis=1)

sns.boxplot(data=df_all, palette='Set3')
plt.ylabel('Density')
plt.show()


# In[28]:


fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

sns.scatterplot(data=dblp_df, x="Density", y="Nodes", ax = axes[0], label = 'dblp')
sns.scatterplot(data=math_sci_df, x="Density", y="Nodes", ax = axes[1], label = 'math_sci')
sns.scatterplot(data=amazon_df, x="Density", y="Nodes", ax = axes[2], label = 'amazon')
sns.scatterplot(data=flickr_df, x="Density", y="Nodes", ax = axes[3], label = 'flickr')
sns.scatterplot(data=youtube_df, x="Density", y="Nodes", ax = axes[4], label = 'youtube')

plt.tight_layout()
plt.show()


# In[31]:


def get_number_of_edges_between_communities(graph, community, name):
    membership = community.membership
    inter_community_edges = 0
    for edge in graph.es:
        if membership[edge.source] != membership[edge.target]:
            inter_community_edges += 1

    print(f"Number of edges between communities for {name}: {inter_community_edges}")


# In[32]:


get_number_of_edges_between_communities(math_sci, math_sci_louvain_communities, "math_sci")
get_number_of_edges_between_communities(dblp, dblp_louvain_communities, "dblp")
get_number_of_edges_between_communities(amazon, amazon_louvain_communities, "amazon")
get_number_of_edges_between_communities(flickr, flickr_louvain_communities, "flickr")
get_number_of_edges_between_communities(youtube, youtube_louvain_communities, "youtube")


# In[ ]:




