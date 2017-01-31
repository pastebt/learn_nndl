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


def dump():
    t, v, s = load_data()
    dump_data('trai_data.txt', t)
    dump_data('vali_data.txt', v)
    dump_data('test_data.txt', s)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.biases = [np.array([[0.1]] * y) for y in sizes[1:]]
        self.biases = [np.reshape(range(1, y + 1), (y, 1)) * 0.1 for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x)
        #self.weights = [np.array([[0.1] * x] * y)
        self.weights = [np.reshape(range(1, x * y + 1), (y, x)) * 0.1
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def test():
    z = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print "sigmoid:"
    print sigmoid(z)
    print "sigmoid_prime:"
    print sigmoid_prime(z)

    nw = Network([4, 3, 2])
    print "feedforward:"
    print nw.feedforward(np.array([[1],
                                   [2],
                                   [3],
                                   [4]]))

    nw = Network([4, 3, 2])
    print "backprop"
    print nw.backprop(np.array([[1],
                                [2],
                                [3],
                                [4]]),
                      np.array([[5],
                                [6]]))

test()
