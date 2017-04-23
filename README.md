 # subgraph2vec

The subgraph2vec paper could be found at: https://arxiv.org/pdf/1606.08928.pdf 

## Virtual Machine with working subgraph2vec setup ##
* An Ubuntu 16.04 vitural machine with all the libraries required for running subgraph2vec could be downloaded from: https://drive.google.com/open?id=0B5UPYObHDu7iWEE0ZlV5WXlhOGs
* Credentials of the VM: username = sg2vec, password = sg2vec
* Just clone this repository and follow the instructions mentioned below to run subgraph2vec on this VM
NOTE: This VM contains a older version of this repository in its home folder. So, please delete it and clone afresh for accessing the latest code. 

#### Dependencies
This code is developed in python 2.7. It is ran and tested on Ubuntu 14.04 and 16.04.
It uses the following python packages:
1. gensim (version == 0.12.4)
2. networkx (version <= 1.11)
3. joblib (version <= 0.10.3)
4. scikit-learn (+scipy, +numpy)

Subgraph2vec is developed on top of "gensim" python package.
We have edited the native word2vec_inner.py file and created a new .so file named 'mod_word2vec_inner.so'.
NOTE: tensorflow version of the same will be released soon.

#####  The procedure for setting up subgraph2vec is as follows:
	1. git clone the repository (command: git clone https://github.com/MLDroid/subgraph2vec.git )
	2. sudo run init.sh (installs gensim, modifies word2vec libraries to use radial skipgram (refer sec: 5.2.2 of the paper))
	NOTE: no need to run init.sh on the sg2vec VM (it is already setup)

#####  The procedure for verifying the setup is as follows:
	1. make sure the gensim package contains a softlink "word2vec_inner.so -> mod_word2vec_inner.so" (command: ls -l /usr/local/lib/python2.7/dist-packages/gensim/models)
	2. make sure that the file named 'Annaword2vec.py' is present in /usr/local/lib/python2.7/dist-packages/gensim/models (command: ls /usr/local/lib/python2.7/dist-packages/gensim/models/Annaword2vec.py)
	3. make sure that the version of gensim is 0.12.4 (command from the python prompt: import gensim;print gensim.__version__)

#####  The procedure for obtaining rooted subgraph vectors using subgraph2vec approach is as follows:
	1. move to the folder "src-code" (command: cd src_code) (also make sure that kdd 2015 paper's (Deep Graph Kernels) datasets are available in '../kdd_datasets/dir_graphs/')
	2. run dump_wl_kernel_sentences.py file to generate the weisfeiler-lehman kernel's rooted subgraphs from all the graphs in a given folder
	   * syntax: python dump_wl_kernel_sentences.py <gexf/json graph_dir> <height of WL kernel> <num of cpu cores for multi-processing>
	   * example: python dump_wl_kernel_sentences.py ../kdd_datasets/dir_graphs/mutag 4 8 (this will generate files with extension .WL4 that contain target and context subgraphs in every graph)
	3. run sg2vec.py to generate embeddings for each of the subgraphs generated in step 2
	   * syntax: python sg2vec.py <src_dir> <file_extension> <opfname_prefix> <embedding_dim> <iterations> <# of cpu cores>
	   * example: python sg2vec.py ../kdd_datasets/dir_graphs/mutag WL4 mutag 32 100 8
	   * this will generate the following two outputs:
	      * i)  a gensim word2vec model containing embeddings for all the subgraphs in the "../models" folder
	      * ii) a json dump of the python dictionary containg embeddings of subgraphs (format of the dictionary: key = subgraph, value = embedding)


## Contact ##
In case of queries, please email: annamala002@e.ntu.edu.sg

#### Reference
	@article{narayanansubgraph2vec,
	  title={subgraph2vec: Learning Distributed Representations of Rooted Sub-graphs from Large Graphs},
	  author={Narayanan, Annamalai and Chandramohan, Mahinthan and Chen, Lihui and Liu, Yang and Saminathan, Santhoshkumar}
	}

