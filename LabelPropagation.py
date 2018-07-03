
# coding: utf-8

# In[11]:


from scipy.sparse import dok_matrix, dia_matrix, identity
from scipy.sparse.linalg import spsolve
import numpy as np
import math
import scipy


# In[1]:


# ネットワークの構造
# ノードiとノードjの間にリンクがあるときW[i,j]=1
W = dok_matrix((10,10))
W[0,2] = W[2,0] = 1
W[0,4] = W[4,0] = 1
W[0,9] = W[9,0] = 1
W[1,2] = W[2,1] = 1
W[2,3] = W[3,2] = 1
W[3,4] = W[4,3] = 1
W[3,6] = W[6,3] = 1
W[5,9] = W[9,5] = 1
W[6,7] = W[7,6] = 1
W[6,8] = W[8,6] = 1
W[7,9] = W[9,7] = 1

# クラスラベル
# 与えられていないときは0
y = dok_matrix([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]]).T

A = W.T * W 

# 単位行列
I = identity(10)

D = dia_matrix((A.sum(0), [0]), (10,10)).tocsr()
D = scipy.sparse.diags(numpy.reciprocal(numpy.sqrt(D).data))
L = I - D*A*D
print(L)
lamb = 0.0001


# In[22]:


#[ 0.45966011  0.23023256  0.46046512  0.1519678   0.5372093  -0.57951699
# -0.38980322 -0.51627907-0.19490161 -0.15903399]
            
f = spsolve((I + (1-lamb) * L), y)


# In[23]:


# top 1, 5, 10, 
f_sort = np.sort(f)

f_desc = f_sort[::-1]
top_1 = f_desc[:1]
top_5 = f_desc[:5]
top_10 = f_desc[:10]

