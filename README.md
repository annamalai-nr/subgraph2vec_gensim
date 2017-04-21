# subgraph2vec

The subgraph2vec paper could be found at: https://arxiv.org/pdf/1606.08928.pdf 

***A working vitural machine with all the code/libraries could be downloaded from: link***

This code is run and tested on Ubuntu 14.04 and 16.04.
The code is developed using python 2.7 (3.x could be obtained by modifying the same).
It uses the following python packages:
1. gensim (version 0.12.4)
2. networkx (version <= 1.11)
3. joblib (version <= 0.10.3)
4. scikit-learn (+scipy, +numpy)

Subgraph2vec is developed on top of "gensim" python package.
We have edited the native word2vec_inner.py file and created a new .so file named 'mod_word2vec_inner.so'.

The procedure for running subgraph2vec to get the embeddings of rooted subgraphs is presented below:

1. git clone the repository
2. run init.sh (installs gensim, modifies word2vec libraries to use radial skipgram (refer sec: 5.2.2 of the paper)
3. run  

