#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("/Users/asmitadeshpande/Documents/faithful.csv")
df.head()


# In[11]:


import matplotlib.pyplot as plt
# Plot eruption time versus waiting time
plt.scatter(df['eruptions'], df['waiting'])
plt.xlabel('Eruption time (in minutes)')
plt.ylabel('Waiting time to next eruption (in minutes)')
plt.title('Old Faithful Geyser Eruptions')
plt.show()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Extract the eruption time and waiting time columns
A= np.array(df[['eruptions', 'waiting']])

# Set the number of clusters
k = 2

# Initialize the cluster centers randomly
centers = np.random.uniform(low=A.min(axis=0), high=A.max(axis=0), size=(k, A.shape[1]))

# Loop until convergence
for i in range(10):
    # Assign each data point to its nearest cluster center
    distances = np.linalg.norm(A[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)

    # Update the cluster centers as the mean of the assigned data points
    for j in range(k):
        centers[j] = np.mean(A[labels == j], axis=0)

    # Plot the current cluster assignments and centers
    plt.scatter(A[:, 0], A[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='green', marker='x', s=100)
    plt.title('Iteration {}'.format(i))
    plt.show()

