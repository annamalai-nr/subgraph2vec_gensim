mkdir models
mkdir data
echo "created models and data folder"

tar -xvzf kdd_datasets.tar.gz
excho "untared the dataset provided in the \"Deep Graph Kernels\" paper by Pinar Yanardag and SVN Vishwanathan"

mv src_code/Annaword2vec.py /usr/local/lib/python2.7/dist-packages/gensim/models/Annaword2vec.py
mv src_code/mod_word2vec_inner.so /usr/local/lib/python2.7/dist-packages/gensim/models/mod_word2vec_inner.so
echo "moved Annaword2vec.py and mod_word2vec_inner.so (containing code for radial skip-gram) to /usr/local/lib/python2.7/dist-packages/gensim/models"

cd /usr/local/lib/python2.7/dist-packages/gensim/models/
mv word2vec_inner.so org_word2vec_inner.so
echo "renamed the word2vec_inner.so at /usr/local/lib/python2.7/dist-packages/gensim/models to org_word2vec_inner.so"
ln -s mod_word2vec_inner.so word2vec_inner.so
echo "created softlink to mod_word2vec_inner.so at /usr/local/lib/python2.7/dist-packages/gensim/models - here onwards, word2vec will use \"radial skipgram\""
cd -
