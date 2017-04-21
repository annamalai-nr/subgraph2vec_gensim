import os, random, gensim
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

def check_gensim_version():
    v = gensim.__version__
    if v != '0.12.4':
        print 'incorrect version of gensim found: {}'.format(v)
        print 'please change the version to gensim version - 0.12.4 and ' \
              'run again (check init.sh for details)'
        exit (-1)
    else:
        print 'correct version of gensim (i.e., 0.12.4) found!'


def main (debug=False):

    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print 'commandline args: <src_dir> <file_extension> <opfname_prefix>' \
              '<embedding_dim> <iterations> <# of cpu cores> ' \
              '<optional:max_num_files>'
        exit(0)

    wlk_target_contexts_dir = sys.argv[1] #folder containing all the
    # WL kernel sentences of the format: <target subgraph> <context subgraphs 1> <context subgraphs 2> ...
    extn = sys.argv[2] #can be WL1, WL2, etc.
    opfname_prefix = sys.argv[3] # could be used as a prefix for the output
    # gensim model - may be used to denote the dataset for which the model is
    # prepared
    embedding_dim = int(sys.argv[4])
    iters = int(sys.argv[5])
    n_cpus = int(sys.argv[6])
    try:
        max_num_files = int(sys.argv[7])
    except:
        max_num_files=0

    #src_dir = '../tmp/MySentencesWithOutDWL1/DrebinWL1/'
    #extn = '.WL4'
    #opfname_prefix = 'ptc'
    #embedding_dim = 32
    #iters = 10
    #n_cpus = 20
    #max_num_files = 0

    check_gensim_version()

    sentences = list(MySentences(wlk_target_contexts_dir, max_num_files, extn))

    neg = 20 #num of negative samples for skipgram model -
    #  TUNE ACCORDING TO YOUR EXPERIMENTAL SETTING
    sg = 1

    model = Word2Vec(sentences, 
                     min_count=1,
                     size=embedding_dim,
                     sg=sg, #make sure this is ALWAYS 1, else cbow model
                     # will be used instead of skip gram
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

    opfname = os.path.join ('../models', opfname_prefix + '_' + '_'.
                            join(['numfiles',
                                   str(max_num_files),
                                   'embeddingdim',
                                   str(embedding_dim),
                                   'numnegsamples',
                                   str(neg),
                                   'numiteration',
                                   str(iters)]))
    model.save(opfname)

    print
    print 'output gensim model containing all the subgraphs and their {} ' \
          'dimensional emeddings for the files from folder {} is saved at {}'\
        .format(embedding_dim, wlk_target_contexts_dir, opfname)
    print
    print 'for loading the embeddings and exploring the gensim model,' \
          'check tutorial at: https://rare-technologies.com/word2vec-tutorial'


if __name__ == '__main__':
    main (debug=False)