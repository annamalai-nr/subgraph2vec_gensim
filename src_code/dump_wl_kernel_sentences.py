import networkx as nx
import os,sys,json, multiprocessing as mp
from time import time
from networkx.readwrite import json_graph
from joblib import Parallel,delayed

def read_from_json_gexf(fname=None,label_field_name='Label',conv_undir = False):
    if not fname:
        print 'no valid path or file name'
        return None
    else:
        try:
            try:
                with open(fname, 'rb') as File:
                    org_dep_g = json_graph.node_link_graph(json.load(File))
            except:
                org_dep_g = nx.read_gexf (path=fname)

            g = nx.DiGraph()
            for n, d in org_dep_g.nodes_iter(data=True):
                g.add_node(n, attr_dict={'label': '-'.join(d[label_field_name].split('\n'))})
            g.add_edges_from(org_dep_g.edges_iter())
        except:
            print "unable to load graph from file", fname
            return 0
    print 'loaded {} a graph with {} nodes and {} egdes'.format(fname,g.number_of_nodes(),g.number_of_edges())
    if conv_undir:
        g = nx.Graph (g)
        print 'converted {} as undirected graph'.format (g)
    return g


def get_graph_as_bow (g, h):
    for n,d in g.nodes_iter(data=True):
        for i in xrange(0, h+1):
            center = d['relabel'][i]
            neis_labels_prev_deg = []
            neis_labels_next_deg = []

            if -1 != i-1:
                neis_labels_prev_deg = [g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g, n)]
            NeisLabelsSameDeg = [g.node[nei]['relabel'][i] for nei in nx.all_neighbors(g,n)]
            if not i+1 > h:
                neis_labels_next_deg = [g.node[nei]['relabel'][i+1] for nei in nx.all_neighbors(g,n)]

            nei_list = NeisLabelsSameDeg + neis_labels_prev_deg + neis_labels_next_deg
            nei_list = ' '.join (nei_list)

            sentence = center + ' ' + nei_list
            yield sentence


def dump_g_as_bow_infile (g,opfname, h):
    Sentences = get_graph_as_bow(g, h)
    with open(opfname, 'w') as fh:
        for w in Sentences:
            print >>fh, w

def wlk_relabel(g,h):
    for n in g.nodes_iter():
        g.node[n]['relabel'] = {}

    for i in xrange(0,h+1): #xrange returns [min,max)
        for n in g.nodes_iter():
            # degree_prefix = 'D' + str(i)
            degree_prefix = ''
            if 0 == i:
                g.node[n]['relabel'][0] = degree_prefix + str(g.node[n]['label']).strip() + degree_prefix
            else:
                nei_labels = [g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g,n)]
                nei_labels.sort()
                sorted_nei_labels = (','*i).join(nei_labels)

                current_in_relabel = g.node[n]['relabel'][i-1] +'#'*i+ sorted_nei_labels
                g.node[n]['relabel'][i] = degree_prefix + current_in_relabel.strip() + degree_prefix

    return g #relabled graph


def process_single_fname (f, h):
    T0 = time()
    print 'processing ',f
    g = read_from_json_gexf(f)
    if not g:
        return
    g = wlk_relabel(g,h)
    dump_g_as_bow_infile (g,opfname=f+'.WL'+str(h), h=h)
    print 'dumped wlk file in {} sec'.format(round(time()-T0,2))

def get_files_to_process(dirname, extn):
    files_to_process = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                files_to_process.append(os.path.join(root, f))

    files_to_process = list(set(files_to_process))
    files_to_process.sort()
    return files_to_process

if __name__ == '__main__':
    if sys.argv[1] in ['-h','--help']:
        print 'command line args: <gexf/json graph_dir> <num of cpu cores for multi-processing> <height of WL kernel> '
        exit (0)

    graph_dir = sys.argv[1] #folder containing the graph's gexf/json format files
    n_cpus = int (sys.argv[2]) #number of cpus to be used for multiprocessing
    h = int (sys.argv[3]) #height of WL kernel (i.e., degree of neighbourhood to consdider)
    extn = '.gexf'

    files_to_process = get_files_to_process(dirname = graph_dir, extn = extn)

    raw_input('have to procees a total of {} files with {} parallel processes... hit any key to proceed...'.
              format(len(files_to_process), n_cpus))

    Parallel(n_jobs=n_cpus)(delayed(process_single_fname)(f, h) for f in files_to_process)


