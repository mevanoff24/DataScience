# Simple Neural Network 


import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from data_helpers2 import load_data



def error_rate(y_true, y_pred):
	return np.mean(y_true != y_pred)

def init_weight_and_bias(input_size, output_size):
	W = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
	b = np.zeros(output_size)
	return W.astype(np.float32), b.astype(np.float32)


class HiddenLayer(object):
	def __init__(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		W, b = init_weight_and_bias(input_size, output_size)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=100, batch_sz=100, show_fig=False):

		num_classes = 2
		X = X.astype(np.float32)
		y = y.astype(np.float32)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1000, random_state = 100)

		N, D = X_train.shape
		self.hidden_layers = []
		input_size = D
		for output_size in self.hidden_layer_sizes:
			h = HiddenLayer(input_size, output_size)
			self.hidden_layers.append(h)
			input_size = output_size
		W, b = init_weight_and_bias(input_size, num_classes)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		self.params = [self.W, self.b]
		for h in self.hidden_layers:
			self.params += h.params


		X = tf.placeholder(tf.float32, [None, D], name = 'X')
		y = tf.placeholder(tf.float32, [None, num_classes], name = 'y')

		pred = self.forward(X)

		rcost = reg * sum([tf.nn.l2_loss(p) for p in self.params])
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, pred)) + rcost

		optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = decay, momentum = mu).minimize(cost)

		prediction = self.predict(X)

		n_batches = N / batch_sz

		costs = []
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(epochs):

				for i in range(n_batches):
					batch_x = X_train[i * batch_sz : (i * batch_sz + batch_sz)]
					batch_y = y_train[i * batch_sz : (i * batch_sz + batch_sz)]

					sess.run(optimizer, feed_dict = {X: batch_x, y: batch_y})

					if i % 20 == 0:
						c, p = sess.run([cost, prediction], feed_dict = {X: X_test, y: y_test})
						costs.append(c)
						print "epoch:", epoch, "i:", i, "nb:", n_batches, "cost:", c, "error rate:",
		
		if show_fig:
			plt.plot(costs)
			plt.show()


	def forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X):
		pred = self.forward(X)
		return tf.argmax(pred, 1)


if __name__ == '__main__':
	X, y, vocabulary, vocabulary_inv = load_data()
	model = ANN([100, 200])
	model.fit(X, y)







