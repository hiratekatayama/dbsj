#!/usr/bin/env python
# coding: utf-8

# ## LiNGAMコード

# In[1]:


import itertools
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment

class LiNGAM():
    def __init__(self,epsilon=1e-25):
        self.epsilon      = epsilon

    def fit(self, X, use_sklearn=False,print_result=True,n_iter=1000,random_state=0):
        self.random_state = random_state
        self.print_result = print_result
        self.n_iter       = n_iter
        self.n_samples, self.n_dim  = X.shape
        X_np = self._pd2np(X)
        #return X_np
        self.X_center           = self._centerize(X_np)
        self.PDW                = self._calc_PDW(use_sklearn=use_sklearn)
        self.P_hat              = self._P_hat()
        self.D_hat,self.DW      = self._PW()
        self.B_hat              = self._B_hat()
        self.P_dot              = self._P_dot()
        self.B_prune            = self._B_prune()
        self.B                  = self._regression_B(X_np)
        self.result_print()
        return self.B

    #if X is pandas DataFrame, convert numpy
    def _pd2np(self,X):
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(self.n_dim)]
        return X_np

    #centerize X by X's col
    def _centerize(self,X):
        return X - np.mean(X,axis=0)

    #whitening using Eigenvalue decomposition
    def _whitening(self,X):
        E, D, E_t = np.linalg.svd(np.cov(X, rowvar=0, bias=0), full_matrices=True)
        ##変えなきゃいけない
        D = np.diag(D**(-1/2))
        V = E.dot(D).dot(E_t) #whitening matrix
        return V.dot(X.T),V
    """
    #whitening using Eigenvalue decomposition
    def _old_whitening(self,X):
        eigen, E = np.linalg.eig(np.cov(X, rowvar=0, bias=0))
        #eigen
        eigen[eigen<0] = -eigen[eigen<0]
        D = np.diag(eigen**(-1/2))
        V = E.dot(D).dot(E.T) #whitening matrix
        return V.dot(X.T),V
    """

    #Estimate W of Wz = s
    def _ICA(self,z,max_iter):
        np.random.seed(self.random_state)
        W_init = np.random.uniform(size=[self.n_dim,self.n_dim])
        W = np.zeros(W_init.shape)
        for i in range(self.n_dim):
            W[i,:] = self._calc_w(W_init[i,:], W, z, max_iter, i)
        return W

    #Estimate PDW
    def _PDW(self,W,V):
        A_tilde = np.linalg.inv(W)
        A = np.linalg.inv(V).dot(A_tilde)
        PDW = np.linalg.inv(A)
        return PDW

    #Estimate P
    def _P_hat(self):
        self.PDW[self.PDW == 0] = self.epsilon
        row_ind, col_ind = linear_sum_assignment(1/np.abs(self.PDW))
        P = np.zeros((len(row_ind),len(col_ind)))
        for i,j in  zip(row_ind,col_ind):
            P[i,j] = 1
        return P

    #Estimate D and DW
    def _PW(self):
        DW = self.P_hat.dot(self.PDW)
        return np.diag(np.diag(DW)),DW

    #Estimate W and B
    def _B_hat(self):
        W_hat = np.linalg.inv(self.D_hat).dot(self.DW)
        B_hat = np.eye(len(W_hat))-W_hat
        return B_hat

    #Estimate P (permute B by causal order)
    def _P_dot(self):
        P_dot_lists = self._get_P_dot_lists()
        score = [self._calc_PBP_upper(P_dot, self.B_hat) for P_dot in P_dot_lists]
        return P_dot_lists[np.argmin(score)]

    #Prune B
    def _B_prune(self):
        B_prune = self.P_dot.dot(self.B_hat).dot(self.P_dot.T)
        for i in range(self.n_dim):
            for j in range(i,self.n_dim):
                B_prune[i,j] = 0
        return self.P_dot.T.dot(B_prune).dot(self.P_dot)

    #Peplace B values with Regression coef
    def _regression_B(self,X):
        causal_matrix = self.B_prune.copy()
        reg_list = {i:causal_matrix[i,:] != 0 for i in range(self.n_dim)}
        for i in range(self.n_dim):
            if np.sum(reg_list[i]) != 0:
                y_reg = X[:,i]
                X_reg = X.T[reg_list[i]].T
                clf = LinearRegression()
                clf.fit(y=y_reg.reshape(self.n_samples,-1), X=X_reg.reshape(self.n_samples,-1))
                causal_matrix[i,reg_list[i]] = clf.coef_
        return causal_matrix

    #FastICA updates
    def _ICA_update(self,w,z):
        w = z.dot((w.T.dot(z)**3)) - 3*w
        w = w/np.sqrt(np.dot(w,w))
        return w

    #calculate w
    def _calc_w(self,w_init,W,z,max_iter,i):
        w_t_1  = w_init.copy()
        W_copy = W.copy()
        for iter_time in range(max_iter):
            w_t = self._ICA_update(w_t_1,z)
            #w_list.append(np.abs(np.dot(w_t,w_t_1)-1))
            if (np.abs(np.dot(w_t,w_t_1)-1) < self.epsilon) or (iter_time == (max_iter-1)):
                #without orthogonalization
                if i==0:
                    return w_t
                #orthogonalization
                else:
                    W_copy[i,:] = w_t
                    w_t = self._calc_gs(W=W_copy,i=i)
                    if (np.abs(np.dot(w_t,w_t_1)-1) < self.epsilon) or (iter_time == (max_iter-1)):
                        return w_t
                    else:
                        w_t_1 = w_t
            else:
                w_t_1 = w_t

    #Estimate W using Sklearn FastICA
    def _W_sklearn(self,X):
        A = FastICA(n_components=self.n_dim).fit(X).mixing_
        return np.linalg.inv(A)

    def _calc_PDW(self,use_sklearn):
        #use FastICA(kurtosis)
        if not use_sklearn:
            z, V   = self._whitening(self.X_center)
            W_z    = self._ICA(z,self.n_iter)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            PDW    = self._PDW(W_z, V)
        #use sklearn's FastICA(neg entropy)
        else:
            PDW    = self._W_sklearn(self.X_center)
        return PDW

    #GS orthogonalization
    def _calc_gs(self,W,i):
        w_i = W[i,:]
        w_add = np.zeros(w_i.shape)
        for j in range(i):
            w_j = W[j:(j+1),:].ravel()
            w_add = w_add + np.dot(w_i,w_j)*w_j
        w_i = w_i - w_add/i
        return w_i/np.sqrt(np.dot(w_i,w_i))

    #get sum of upper triangle value
    def _get_upper_triangle(self,mat):
        return np.diag(mat.dot(np.tri(self.n_dim))).sum()

    #P_dot
    def _get_P_dot_lists(self):
        base_array  = np.eye(N=1,M=self.n_dim).ravel().astype("int")
        base_array  = set(itertools.permutations(base_array))
        return np.array(list(itertools.permutations(base_array)))

    #get PBP to minimize upper triangle value
    def _calc_PBP_upper(self,P_dot,B_hat):
        return self._get_upper_triangle( P_dot.dot(B_hat).dot(P_dot.T)**2)

    #print result
    def result_print(self):
        if self.print_result:
            for i,b in enumerate(self.columns):
                for j,a in enumerate(self.columns):
                    if self.B[i,j]!=0:
                        print(a,"---|%.3f|--->"%(self.B[i,j]),b)


# ## データ読み込み

# In[44]:


import csv

with open("some.csv","r") as f:
    reader = csv.reader(f)
    header = next(reader)


# In[45]:


with open("name.csv","r") as f:
    reader = csv.reader(f)
    name_header = next(reader)


# In[46]:


import numpy as np
new_data = pd.read_csv("zenkoku.csv",engine="python",encoding="utf-8")
X = new_data.drop("Unnamed: 0",axis=1)
col = list(X.columns)


# In[47]:


from sklearn import preprocessing

mm = preprocessing.MinMaxScaler()
X = mm.fit_transform(X)
X = pd.DataFrame(X)
X.columns = col


# In[48]:


X = pd.DataFrame(X)
X.columns = col


# ## LiNGAM実行

# In[50]:


lingam = LiNGAM()
lingam.fit(X)


# In[1]:


from graphviz import Digraph
import pandas as pd


# In[2]:


top_9 = pd.read_csv("top_9.csv",encoding="shift-jis",header=None)


# In[3]:


end = []
for i in range(len(top_9)):
    top = []
    top.append(top_9[0][i].split("---|")[0])
    top.append(top_9[0][i].split("---|")[1].split("|--->")[0])
    top.append(top_9[0][i].split("---|")[1].split("|--->")[1])
    end.append(top)


# In[4]:


for i in range(len(end)):
    for j in range(len(end[0])):
        end[i][j] = end[i][j].replace(" ","")


# In[5]:


from graphviz import Digraph

G = Digraph(format="png")
G.attr("node", shape="square", style="filled")
for i in range(len(end)):
    G.edge(end[i][0], end[i][2], label=end[i][1])
G.node("人口増減率", shape="circle", color="pink")
G.render("zenkoku")


# ### 付録

# In[12]:


lingam = LiNGAM()
lingam.fit(X,use_sklearn=True)

