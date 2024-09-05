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
math_sci = utils.create_graph(math_sci_name)
utils.basic_statistics(math_sci, "math_sci")


# In[13]:


get_ipython().run_cell_magic('timeit', '', '%memit  math_sci_louvain_communities = math_sci.community_multilevel()')


# In[4]:


amazon_name = "com-amazon.ungraph.txt"
amazon = utils.create_graph(amazon_name)
utils.basic_statistics(amazon, "amazon")


# In[11]:


get_ipython().run_cell_magic('timeit', '', '%memit amazon_louvain_communities = amazon.community_multilevel()')


# In[6]:


youtube_name = "com-youtube.ungraph.txt"
youtube = utils.create_graph(youtube_name)
utils.basic_statistics(youtube, "youtube")


# In[12]:


get_ipython().run_cell_magic('timeit', '', '%memit youtube_louvain_communities = youtube.community_multilevel()')


# In[8]:


dblp_name = "com-dblp.ungraph.txt"
dblp = utils.create_graph(dblp_name)
utils.basic_statistics(dblp, "dblp")


# In[9]:


get_ipython().run_cell_magic('timeit', '', '%memit dblp_louvain_communities = dblp.community_multilevel()')


# In[19]:


flickr_name = "soc-flickr.mtx"
flickr = utils.create_graph(flickr_name)
utils.basic_statistics(flickr, "flickr")


# In[20]:


get_ipython().run_cell_magic('timeit', '', '%memit flickr_louvain_communities = flickr.community_multilevel()')


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

nodes = [317080, 332689, 334863, 513969, 1134890]
times = [9.44, 15.3, 5.43, 26.4, 24.8]
errors = [1.86, 3.68, 0.485, 7.65, 3.12]
networks = ['dblp', 'math_sci', 'amazon', 'flickr', 'youtube']

plt.errorbar(nodes, times, yerr=errors, fmt='-o', label='Time')

# Add labels for each node
for i, txt in enumerate(networks):
    plt.text(nodes[i] + 60000, times[i], txt, ha='right', va='bottom')

x = np.linspace(min(nodes), max(nodes), 100)
y = 0.0000020 * x * np.log(x) - 8

plt.plot(x, y, label='O(n log n)', linestyle='--', color='red')
plt.xlabel('Number of nodes')
plt.ylabel('Louvain Run Time (s)')

plt.legend()

plt.show()


# In[ ]:




