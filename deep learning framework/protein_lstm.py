# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np

# master = data_input.data_master()


time_steps = 12
channel_size = 3
embedding_size = 64
loc_size = 11

embedding_fn_size = 312
embedding_loc_size = 512

# lstm_cell_size = 8
lstm_cell_size = 8

threshold = 0.5


class Model(object):
    def __init__(self, init_learning_rate, decay_steps, decay_rate):
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate,
                                                   staircase=True)

        # define placehold
        self.x = tf.placeholder(tf.float32, [None, channel_size, time_steps])
        x_emb = tf.transpose(self.x, [0, 2, 1])  # [None,time_steps,channel_size]

        self.e = tf.placeholder(tf.float32, [None, embedding_size])
        self.l = tf.placeholder(tf.float32, [None, loc_size])
        self.y = tf.placeholder(tf.int32, [None, 1])

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ones = tf.ones_like(self.y)
        zeros = tf.zeros_like(self.y)

        with tf.name_scope("FN_Part"):
            output_e = tf.layers.dense(self.e, embedding_fn_size, activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            output_l = tf.layers.dense(self.l, embedding_loc_size, activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        with tf.name_scope("LSTM_Part"):
            output_lstm = self.BidirectionalLSTMEncoder(x_emb)
            print('after BidirectionalLSTMEncoder: ', output_lstm)
            output_lstm = tf.reshape(output_lstm, [-1, 2 * lstm_cell_size])

            print(output_lstm)

        with tf.name_scope("Output_Part"):
            # concate 3 part
            concate_v = tf.concat([output_e, output_l, output_lstm], axis=1)  # our_method
            # concate 2 part
            # concate_v = tf.concat([output_l, output_lstm], axis=1) # without_PPI_network
            # concate_v = tf.concat([output_e, output_lstm], axis=1)  # without_localization
            # concate_v = tf.concat([output_e, output_l], axis=1)  # without_expression
            # input_size = concate_v.get_shape().as_list()[-1]
            input_size = 2 * lstm_cell_size + embedding_fn_size + embedding_loc_size

            weight_last = tf.Variable(
                tf.truncated_normal([input_size, 1]) * np.sqrt(
                    2. / (3 * lstm_cell_size)))
            bias_last = tf.Variable(tf.truncated_normal([1], stddev=0.1))
            concate_v = tf.nn.dropout(concate_v, self.dropout_keep_prob)
            logits_cnn = tf.matmul(concate_v, weight_last) + bias_last

            self.loss_cnn = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32), logits=logits_cnn))
            self.optimizer_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_cnn,
                                                                                              global_step=global_step)
            self.logits_pred = tf.nn.sigmoid(logits_cnn)
            self.prediction_cnn = tf.cast(tf.where(tf.greater(self.logits_pred, threshold), ones, zeros), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction_cnn, self.y), tf.float32))

    def BidirectionalLSTMEncoder(self, inputs, name='BidirectionalLSTM'):
        # 输入inputs的shape是[batch_size*sent_in_doc, word_in_sent, embedding_size]
        print(inputs)
        with tf.variable_scope(name):
            LSTM_cell_fw = rnn.LSTMCell(lstm_cell_size)
            LSTM_cell_bw = rnn.LSTMCell(lstm_cell_size)

            LSTM_cell_fw = rnn.DropoutWrapper(LSTM_cell_fw, output_keep_prob=self.dropout_keep_prob)
            LSTM_cell_bw = rnn.DropoutWrapper(LSTM_cell_bw, output_keep_prob=self.dropout_keep_prob)
            # fw_outputs和bw_outputs的size都是[batch_size*sent_in_doc, word_in_sent, embedding_size]
            #  tuple of (outputs, output_states)
            ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_cell_fw,
                                                                                               cell_bw=LSTM_cell_bw,
                                                                                               inputs=inputs,
                                                                                               sequence_length=self.length(
                                                                                                   inputs),
                                                                                               dtype=tf.float32)
            # outputs的size是[batch_size, hidden_size*2]
            # outputs = tf.concat((fw_outputs[:, -1, :], bw_outputs[:, -1, :]), 1)
            outputs = tf.concat((fw_state.h, bw_state.h), 1)
            return outputs

    def length(self, sequences):
        # 动态展开
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        self.seq_len = tf.cast(seq_len, tf.int32)
        return self.seq_len
