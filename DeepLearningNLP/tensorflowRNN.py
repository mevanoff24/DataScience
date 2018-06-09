from data_helpers2 import *
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
# from tensorflow.python.ops import rnn as rnn_module
from tensorflow.python.ops.rnn import rnn as get_rnn_output
from tensorflow.python.ops.rnn_cell import BasicRNNCell, GRUCell
from sklearn.utils import shuffle
# from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

class SimpleRNN:
  def __init__(self, M):
    self.M = M # hidden layer size


  def fit(self, X, Y, batch_sz=20, learning_rate=10e-1, mu=0.99, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
    D = 1
    N, T = X.shape # X is of size N x T(n) x D
    K = len(set(Y.flatten()))
    M = self.M
    self.f = activation

    # initial weights
    # note: Wx, Wh, bh are all part of the RNN unit and will be created
    #       by BasicRNNCell
    Wo = init_weight(M, K).astype(np.float32)
    bo = np.zeros(K, dtype=np.float32)

    # make them tf variables
    self.Wo = tf.Variable(Wo)
    self.bo = tf.Variable(bo)

    # tf Graph input
    tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
    tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')

    # turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
    sequenceX = x2sequence(tfX, T, D, batch_sz)

    # create the simple rnn unit
    rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

    # Get rnn cell output
    # outputs, states = rnn_module.rnn(rnn_unit, sequenceX, dtype=tf.float32)
    outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

    # outputs are now of size (T, batch_sz, M)
    # so make it (batch_sz, T, M)
    outputs = tf.transpose(outputs, (1, 0, 2))
    outputs = tf.reshape(outputs, (T*batch_sz, M))

    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(outputs, self.Wo) + self.bo
    predict_op = tf.argmax(logits, 1)
    targets = tf.reshape(tfY, (T*batch_sz,))

    cost_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets))
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)

    costs = []
    n_batches = N / batch_sz
    
    init = tf.initialize_all_variables()
    with tf.Session() as session:
      session.run(init)
      for i in xrange(epochs):
        X, Y = shuffle(X, Y)
        n_correct = 0
        cost = 0
        for j in xrange(n_batches):
          Xbatch = X[j*batch_sz:(j+1)*batch_sz]
          Ybatch = Y[j*batch_sz:(j+1)*batch_sz]
          
          _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
          cost += c
          for b in xrange(batch_sz):
            idx = (b + 1)*T - 1
            n_correct += (p[idx] == Ybatch[b][-1])
        if i % 10 == 0:
          print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
        if n_correct == N:
          print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
          break
        costs.append(cost)

    if show_fig:
      plt.plot(costs)
      plt.show()




if __name__ == '__main__':
    X, y, vocabulary, vocabulary_inv = load_data()
    rnn = SimpleRNN(4)
rnn.fit(X, y,
    batch_sz=10,
    learning_rate=0.001,
    epochs=2,
    activation=tf.nn.sigmoid,
    show_fig=False
  )