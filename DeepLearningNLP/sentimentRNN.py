# Simple Recurrent Neural Network 


import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_helpers2
from sklearn.model_selection import train_test_split



X, y, vocabulary, vocabulary_inv = data_helpers2.load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

N, D = X_train.shape

hm_epochs = 100
n_classes = 2
batch_size = 128
dimensions = 1
n_chunks = D
rnn_size = 128

num_batches = N / batch_size


x = tf.placeholder('float', [None, n_chunks,dimensions])
y = tf.placeholder('float', [None, n_classes])


def accuracy(y_true, y_pred):
	return np.mean(y_true == y_pred)


def recurrent_neural_network(x):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, dimensions])
	x = tf.split(0, n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

	return output


def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for i in range(num_batches):
				batch_x = X_train[i * batch_size : (i * batch_size + batch_size)]
				batch_x = batch_x.reshape((batch_size,n_chunks,dimensions))
				batch_y = y_train[i * batch_size : (i * batch_size + batch_size)]
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)




train_neural_network(x)