import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

nb_users = 943
nb_items = 1682


with open("data/ml-100k/u.data") as tsv:
    data = [[ int(item) for item in line.strip().split('\t')] for line in tsv]

matrixTab = np.zeros((nb_users+1, nb_items+1))

for line in data:
    matrixTab[line[0],line[1]] = line[2]

mt_sparse = sparse.csr_matrix(matrixTab)

uu_similarities = cosine_similarity(mt_sparse)
#print('user-user similarity:\n {}\n'.format(uu_similarities))

ii_similarities = cosine_similarity(mt_sparse.transpose())
#print('item-item similarity:\n {}\n'.format(ii_similarities))

print(ii_similarities.shape)

print(np.where(uu_similarities[1] < 1.)[0])
