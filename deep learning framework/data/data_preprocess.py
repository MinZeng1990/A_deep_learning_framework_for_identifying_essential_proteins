# encoding=utf-8

import pickle
import numpy as np

embedding_size = 64
loc_size = 11

with open('../data/biogrid_dict.pkl', 'rb') as file:
    biogrid_dict = pickle.load(file)
biogrid_dict = {value: key for key, value in biogrid_dict.items()}

# topology node2vec embedding information
emb_dict = dict()
with open('../data/biogrid_network_(p2q1).emb') as file:
    for line in file.readlines()[1:]:
        columns = line.strip().split()
        protein_id = columns[0]
        protein_emb = columns[1:]
        assert len(protein_emb) == 64
        protein_name = biogrid_dict[protein_id]
        emb_dict[protein_name] = protein_emb

# essential protein information
key_protein_set = set()
with open('../data/essential_proteins.txt') as file:
    for line in file.readlines():
        key_protein = line.strip()
        key_protein_set.add(key_protein)

# subcellular information
with open('../data/protein_subcellular_dict.pkl', 'rb') as file:
    protein_dict = pickle.load(file)

protein_matrix = []
protein_label = []
protein_emb = []
protein_loc = []
# gene expression information
with open('../data/GSE3431_T36.txt') as file:
    for line in file.readlines():
        columns = line.strip().split()
        protein_name = columns[0]
        features = columns[1:]
        assert len(features) == 36

        if protein_name in emb_dict:
            protein_matrix.append(features)
            protein_emb.append(emb_dict[protein_name])
            protein_loc.append(protein_dict.get(protein_name, [0] * loc_size))
            if protein_name in key_protein_set:
                protein_label.append(1)  # 1266
            else:
                protein_label.append(0)  # 5510

protein_matrix = np.array(protein_matrix, dtype=np.float32).reshape(-1, 3, 12)
np.save('../data/protein_matrix.npy', protein_matrix)
print(protein_matrix.shape)

protein_emb = np.array(protein_emb, dtype=np.float32)
np.save('../data/protein_emb.npy', protein_emb)
print(protein_emb.shape)

protein_loc = np.array(protein_loc, dtype=np.float32)
np.save('../data/protein_loc.npy', protein_loc)
print(protein_loc.shape)

protein_label = np.array(protein_label, dtype=np.int32).reshape(-1, 1)
np.save('../data/protein_label.npy', protein_label)
print(protein_label.shape)
print(sum(protein_label))
