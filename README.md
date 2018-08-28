# A_deep_learning_framework_for_identifying_essential_proteins
A deep learning framework for identifying essential proteins by integrating multiple sources of biological information 
# Requirements

tensorflow==1.0.0

numpy==1.11.2

networkx==2.0

scikit-learn==0.18

gensim==0.13.3

# Usage

  Please follow the instructions of node2vec ( https://github.com/aditya-grover/node2vec ) to generate the node vectors of PPI network. In our study, we generated 64-dimensional vector for each node. The specific parameters are as follows. Length of walk per source is 20, number of walk per source is 10, window context size for optimization is 10, p is 2, and the rest parameters remain the default settings. 
  
  In this GitHub project, we give a demo to show how it works. We give the three datasets.   
  
  1. protein_emb.npy is the 64-dimensional vector which is generated by node2vec by using PPI network. Its shape is 5297 proteins x 64 features.

  2. protein_matrix.npy is the gene expression data.

  3. protein_matrix.npy is the labels of proteins (1:essential proteins and 0: non-essential proteins).

  You can split the raw dataset by yourself. In our demo, we use the 80% as training dataset and 20% as testing dataset. The detail of dataset division can see the paper and the code.
 
  You can run the main function to see the resluts and predict the essential proteins.
 
# Citation

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
