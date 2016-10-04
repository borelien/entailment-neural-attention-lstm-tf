import tensorflow as tf
import numpy as np

class TensorFlowTrainable(object):
    def __init__(self):
        self.parameters = []

    def get_weights(self, dim_in, dim_out, name, trainable=True):
        shape = (dim_out, dim_in)
        weightsInitializer = tf.constant_initializer(self.truncated_normal(shape=shape, stddev=0.01, mean=0.))
        weights = tf.get_variable(initializer=weightsInitializer, shape=shape, trainable=True, name=name)
        if trainable:
            self.parameters.append(weights)
        return weights

    def get_4Dweights(self, filter_height, filter_width, in_channels, out_channels, name, trainable=True):
        shape = (filter_height, filter_width, in_channels, out_channels)
        weightsInitializer = tf.constant_initializer(self.truncated_normal(shape=shape, stddev=0.01, mean=0.))
        weights = tf.get_variable(initializer=weightsInitializer, shape=shape, trainable=True, name=name)
        if trainable:
            self.parameters.append(weights)
        return weights

    def get_biases(self, dim_out, name, trainable=True):
        shape = (dim_out, 1)
        initialBiases = tf.constant_initializer(np.zeros(shape))
        biases = tf.get_variable(initializer=initialBiases, shape=shape, trainable=True, name=name)
        if trainable:
            self.parameters.append(biases)
        return biases

    @staticmethod
    def truncated_normal(shape, stddev, mean=0.):
        rand_init = np.random.normal(loc=mean, scale=stddev, size=shape)
        inf_mask = rand_init < (mean - 2 * stddev)
        rand_init = rand_init * np.abs(1 - inf_mask) + inf_mask * (mean - 2 * stddev)
        sup_mask = rand_init > (mean + 2 * stddev)
        rand_init = rand_init * np.abs(1 - sup_mask) + sup_mask * (mean + 2 * stddev)
        return rand_init

class LSTMCell(TensorFlowTrainable):
    def __init__(self, num_units, **kwargs):
        super(LSTMCell, self).__init__()

        # k
        self._num_units = num_units

        # weights
        self.w_i = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_i")
        self.w_f = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_f")
        self.w_o = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_o")
        self.w_c = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_c")
        
        # biases
        self.b_i = self.get_biases(dim_out=self._num_units, name="b_i")
        self.b_f = self.get_biases(dim_out=self._num_units, name="b_f")
        self.b_o = self.get_biases(dim_out=self._num_units, name="b_o")
        self.b_c = self.get_biases(dim_out=self._num_units, name="b_c")

        # state
        self.c = [self.get_biases(dim_out=self._num_units, name="c", trainable=False)]

    def initialize_something(self, input):
        self.batch_size_vector = 1 + 0 * tf.expand_dims(tf.unpack(tf.transpose(input, [1, 0]))[0], 0)
        self.h = [self.get_biases(dim_out=self._num_units, name="h", trainable=False) * self.batch_size_vector]

    def process(self, input):
        H = tf.concat(0, [tf.transpose(input, perm=[1, 0]), self.h[-1]])
        i = tf.sigmoid(x=tf.add(tf.matmul(self.w_i, H), self.b_i))
        f = tf.sigmoid(x=tf.add(tf.matmul(self.w_f, H), self.b_f))
        o = tf.sigmoid(x=tf.add(tf.matmul(self.w_o, H), self.b_o))
        self.c.append(f * self.c[-1] + i * tf.tanh(x=tf.add(tf.matmul(self.w_c, H), self.b_c)))
        self.h.append(o * tf.tanh(x=self.c[-1]))

    @property
    def features(self):
        return self.h[-1]

class AttentionLSTMCell(LSTMCell):
    def __init__(self, num_units, hiddens, states, **kwargs):
        super(AttentionLSTMCell, self).__init__(num_units=num_units)

        # warm-up
        self.warm_hiddens = hiddens
        self.L = len(self.warm_hiddens)
        self.c = [states[-1]]
        self.h = [hiddens[-1]]

        # weights
        self.w_y = self.get_4Dweights(filter_height=self._num_units, filter_width=1, in_channels=1, out_channels=self._num_units, name="w_y")
        self.w = self.get_4Dweights(filter_height=self._num_units, filter_width=1, in_channels=1, out_channels=1, name="w")
        self.w_h = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_h")
        self.w_r = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_r")
        self.w_t = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_t")
        self.w_p = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_p")
        self.w_x = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_x")

    def initialize_something(self, input):
        super(AttentionLSTMCell, self).initialize_something(input=input)

        # attention
        self.r = [self.get_biases(dim_out=self._num_units, name="r", trainable=False) * self.batch_size_vector]

        # warming-up
        self.Y = tf.expand_dims(tf.transpose(tf.pack(self.warm_hiddens), [2, 1, 0]), 3)

    def process(self, input):
        # classic-LSTM module
        super(AttentionLSTMCell, self).process(input=input)

        # attention-LSTM module
        firs_term = tf.transpose(tf.nn.conv2d(input=self.Y, filter=self.w_y, strides=[1, 1, 1, 1], padding="VALID"), [0, 3, 2, 1])
        second_term = tf.expand_dims(tf.transpose(tf.tile(tf.expand_dims(tf.matmul(self.w_h, self.h[-1]) + tf.matmul(self.w_r, self.r[-1]), [2]), [1, 1, self.L]), [1, 0, 2]), 3)
        M = tf.tanh(firs_term + second_term)
        alpha = tf.expand_dims(tf.nn.softmax(tf.squeeze(tf.nn.conv2d(input=M, filter=self.w, strides=[1, 1, 1, 1], padding="VALID"), [1, 3])), 2)
        self.r.append(tf.transpose(tf.squeeze(tf.batch_matmul(tf.squeeze(self.Y, [3]), alpha), [2]), [1, 0]) + tf.tanh(tf.matmul(self.w_t, self.r[-1])))

    @property
    def features(self):
        return tf.tanh(tf.matmul(self.w_p, self.r[-1]) + tf.tanh(tf.matmul(self.w_x, self.h[-1])))

class RNN(TensorFlowTrainable):
    def __init__(self, cell, num_units, embedding_dim, projecter, keep_prob, **kwargs):
        super(RNN, self).__init__()
        
        # private
        self._projecter = projecter
        self._embedding_dim = embedding_dim
        self._num_units = num_units
        self._cell = cell(num_units=self._num_units, **kwargs)
        self.keep_prob = keep_prob

        # public
        self.predictions = None
        self.hiddens = None
        self.states = None

    def process(self, sequence):
        noisy_sequence = tf.nn.dropout(x=sequence, keep_prob=self.keep_prob, name="noisy_inputs")
        noisy_sequence = tf.expand_dims(tf.transpose(noisy_sequence, [1, 0, 2]), 3)
        projected_sequence = tf.transpose(tf.squeeze(tf.nn.conv2d(input=noisy_sequence, filter=self._projecter, strides=[1, 1, 1, 1], padding="VALID"), [2]), [1, 0, 2])
        
        list_sequence = tf.unpack(projected_sequence)
        self._cell.initialize_something(input=list_sequence[0])
        for i, input in enumerate(list_sequence):
            self._cell.process(input=input)
        self.states, self.hiddens = self._cell.c[1:], self._cell.h[1:]

    def get_predictions(self):
        biases = self.get_biases(dim_out=3, name="biases")
        weights = self.get_weights(dim_in=self._num_units, dim_out=3, name="weights")
        noisy_features = tf.nn.dropout(x=self._cell.features, keep_prob=self.keep_prob, name="noisy_features")
        self.predictions = tf.transpose(tf.add(tf.matmul(weights, noisy_features), biases), [1, 0])
        return self.predictions
    
    def loss(self, targets):
        if self.hiddens is None:
            raise Exception("You shouldn't have been there.")
        else:
            with tf.name_scope("loss") as scope:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.get_predictions(), targets))
                loss_summary = tf.scalar_summary("loss", loss)
            with tf.name_scope("accuracy") as scope:
                predictions = tf.to_int32(tf.argmax(self.predictions, 1))
                correct_label = tf.to_int32(targets)
                correct_predictions = tf.equal(predictions, correct_label)
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
                accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            return loss, loss_summary, accuracy, accuracy_summary
