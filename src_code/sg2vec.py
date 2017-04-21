import os, random
from gensim.models.Annaword2vec import *
# from gensim.models.word2vec import *
from pprint import pprint

class MySentences(object):
    def __init__(self, dirname, num_files, Extn):
        self.dirname = dirname
        self.fnames = []
        if num_files:
            for fname in os.listdir(self.dirname)[:num_files]:
                if fname.endswith(Extn):
                    self.fnames.append(os.path.join(self.dirname, fname))
        else:
            for fname in os.listdir(self.dirname):
                if fname.endswith(Extn):
                    self.fnames.append(os.path.join(self.dirname, fname))
        print 'loaded {} fnames from {}'.format(len(self.fnames), self.dirname)

    def __iter__(self):
        for fname in self.fnames:
            for line in open(fname):
                yield line.split()


def shuffle(sents):
    random.shuffle(sents)
    return sents


def main (debug=False):

    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print 'commandline args: <src_dir> <max_num_files> <file_extension> <opfname_prefix> <embedding_dim> <iterations> <# of cpu cores>'
        exit(0)

    wlk_target_contexts_dir = sys.argv[1]
    max_num_files = int(sys.argv[2])
    extn = sys.argv[3]
    opfname_prefix = sys.argv[4]
    embedding_dim = int(sys.argv[5])
    iters = int(sys.argv[6])
    n_cpus = int(sys.argv[7])

    #src_dir = '../tmp/MySentencesWithOutDWL1/DrebinWL1/'
    #num_files = 1000
    #extn = 'Removed'
    #opfname_prefix = 'subgraph2vec'
    #embedding_dim = 32
    #iters = 10
    #n_cpus = 20


    sentences = list(MySentences(wlk_target_contexts_dir, max_num_files, extn))

    neg = 20 #num of negative samples for skipgram model - CHANGE ACCORDING TO YOUR WISH
    sg = 1

    model = Word2Vec(sentences, 
                     min_count=1,
                     size=embedding_dim,
                     sg=sg,
                     negative=neg,
                     workers=n_cpus,
                     iter=iters)

    if debug:
        vocab = model.vocab.keys()
        for w in vocab[:10]:
            print 'closest subgraphs to: {} are as follows'.format(w)
            print '-' * 80
            pprint(model.most_similar(w))
            print '-' * 80

    opfname = os.path.join ('../models', opfname_prefix + '_' + '_'.join([str(max_num_files),
                                                                          str(embedding_dim),
                                                                          str(sg),
                                                                          str(neg),
                                                                          str(iters)]))
    model.save(opfname)

    print 'output gensim model containing all the subgraphs and their {} ' \
          'dimensional emeddings for the files from folder {} is saved at {}'\
        .format(embedding_dim, wlk_target_contexts_dir, opfname)

    print 'for loading the embeddings and exploring the gensim model,' \
          'check tutorial at: https://rare-technologies.com/word2vec-tutorial'


if __name__ == '__main__':
    main (debug=False)