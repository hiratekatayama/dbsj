
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np
from scipy.sparse import dok_matrix


# In[24]:


a_csv = pd.read_csv('C://Users//hirakata//desktop//joint research//PMI_file//PMI//a.csv', names = ('keyword','url','keyword_url_cnt','key_cnt','url_cnt'))


# In[26]:


key_url = pd.DataFrame(a_csv)


# In[51]:


a = np.log(key_url["keyword_url_cnt"]*10835497)/(key_url["key_cnt"]*key_url["url_cnt"])
b = -np.log(key_url["keyword_url_cnt"]/10835497)
NPMI = a/b
p = pd.DataFrame(NPMI)


# In[55]:


NPMI_new = p.rename(columns={0:"NPMI"})


# In[73]:


a = pd.concat([key_url, NPMI_new], axis = 1, join = "inner")


# In[60]:


Ukey = a["keyword"].unique()
Uurl = a["url"].unique()


# In[64]:


print(len(Ukey),len(Uurl))


# In[61]:


key2index = {}
i = 0

for k in Ukey:
    key2index[k] = i
    i += 1


# In[62]:


url2index = {}
i = 0

for u in Uurl:
    url2index[u] = i
    i += 1


# In[106]:


W = dok_matrix((len(Ukey),len(Uurl)))

for key, column in a.iterrows():
    print((key+1)/len(a))
    W[key2index[column["keyword"]],url2index[column["url"]]] = column["NPMI"]



