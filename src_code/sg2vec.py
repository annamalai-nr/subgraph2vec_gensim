import os, random, gensim, json
from gensim.models.Annaword2vec import *
# from gensim.models.word2vec import *
from pprint import pprint
from time import time

class sg2vec_sentences (object):
    def __init__(self, dirname, num_files, extn):
        '''
        loads all the WL kernel relabeled files with a particular extension from a given folder
        :param dirname: target folder containing WL kernel relabeled files
        :param num_files: maximum number of files to be loaded. if num_files = 0, all the files in the folder will be
        loaded
        :param extn: extension of the files to be loaded (e.g., WL1, WL2, etc.) 
        '''
        self.dirname = dirname
        self.fnames = []
        if num_files:
            for fname in os.listdir(self.dirname)[:num_files]:
                if fname.endswith(extn):
                    self.fnames.append(os.path.join(self.dirname, fname))
        else:
            for fname in os.listdir(self.dirname):
                if fname.endswith(extn):
                    self.fnames.append(os.path.join(self.dirname, fname))
        print 'loaded {} fnames from {}'.format(len(self.fnames), self.dirname)

    def __iter__(self):
        '''
        generator to parse one line from one file at a time. Each line is of the format: 
        <target subgraph> <context subgraph 1> <context subgraph 2> ... <context subgraph n> 
        :return: List of the format [target subgraph, context subgraph 1, context subgraph 2, ... context subgraph n]
        '''
        for fname in self.fnames:
            for line in open(fname):
                yield line.split()


def shuffle(sents):
    '''
    randomizing list of sentences 
    :param sents: input sentences iterbale
    :return: sentences shuffled in random order
    '''
    random.shuffle(sents)
    return sents


def check_gensim_version():
    '''
    Util function to check the version of gensim being imported.
    This code is only tested with gensim version 0.12.4
    :return:None 
    '''
    v = gensim.__version__
    if v != '0.12.4':
        print 'incorrect version of gensim found: {}'.format(v)
        print 'please change the version to gensim version - 0.12.4 and ' \
              'run again (check init.sh for details)'
        exit (-1)
    else:
        print 'correct version of gensim (i.e., 0.12.4) found!'



def main (debug=False, save_subgraph_vec_map= True):
    '''
    main function which reads the command line args and trains the gensim word2vec model to obtain subgraph embeddings
    :param debug: Flag to control whether to print/check vocabulary of subgraphs 
    :param save_subgraph_vec_map: Flag to control whether to save the python dictinoary of the format {subgraph:vector}.
    This dictinoary is saved as a json file if save_subgraph_vec_map is set to True
    :return: None
    '''

    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print 'commandline args: <src_dir> <file_extension> <opfname_prefix> ' \
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

    # wlk_target_contexts_dir = '../malware_dataset/DrebinWL1/'
    # # wlk_target_contexts_dir = '../kdd_datasets/dir_graphs/ptc'
    # extn = 'Removed'
    # # extn = 'WL4'
    # opfname_prefix = 'drebin'
    # embedding_dim = 32
    # iters = 2
    # n_cpus = 20
    # max_num_files = 1000

    check_gensim_version()

    sentences = sg2vec_sentences(wlk_target_contexts_dir, max_num_files, extn)

    neg = 20 #num of negative samples for skipgram model -TUNE ACCORDING TO YOUR EXPERIMENTAL SETTING
    sg = 1

    t0 = time()
    model = Word2Vec(sentences, 
                     min_count=1,
                     size=embedding_dim,
                     sg=sg, #make sure this is ALWAYS 1, else cbow model will be used instead of skip gram
                     negative=neg,
                     workers=n_cpus,
                     iter=iters)
    print 'trained the gensim subgraph2vec model in {} sec.'.format(round(time()-t0,2))

    if debug:
        vocab = model.vocab.keys()
        for w in vocab[:10]:
            print 'closest subgraphs to: {} are as follows'.format(w)
            print '-' * 80
            pprint(model.most_similar(w))
            print '-' * 80

    op_fname = opfname_prefix + '_' + '_'.join(['numfiles',
              str(max_num_files),
              'embeddingdim',
              str(embedding_dim),
              'numnegsamples',
              str(neg),
              'numiteration',
              str(iters)])
    model.save(os.path.join ('../models', op_fname))

    print
    print '-' * 80
    print 'output gensim model containing all the subgraphs and their {} ' \
          'dimensional emeddings for the files from folder {} is saved at {}'\
        .format(embedding_dim, wlk_target_contexts_dir, os.path.join ('../models', op_fname))
    print '-' * 80
    print
    print 'for loading the embeddings and exploring the gensim model,' \
          'check tutorial at: https://rare-technologies.com/word2vec-tutorial'

    t0 = time()
    if save_subgraph_vec_map:
        subgraph_vec_map = {subgraph:model[subgraph].tolist() for subgraph in model.vocab}
        op_fname = op_fname + '_map.json'
        with open (op_fname,'w') as fh:
            json.dump(obj=subgraph_vec_map, fp =fh, indent=4)

    print
    print '-' * 80
    print 'saved the subgraph:vector map in file {} in {} sec.'.format(op_fname, round(time() - t0, 2))
    print '-' * 80


if __name__ == '__main__':
    main (debug=False)