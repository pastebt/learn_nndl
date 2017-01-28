# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('../neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def dump_data(outfn, data):
    #print data[0]
    #print data[1]
    fout = open(outfn, 'w')
    for i, n in enumerate(data[1]):
        #print n, data[0][i]
        fout.write("%d\t" % n)
        fout.write('\t'.join(map(str, data[0][i])))
        fout.write('\n')


t, v, s = load_data()
dump_data('trai_data.txt', t)
dump_data('vali_data.txt', v)
dump_data('test_data.txt', s)
