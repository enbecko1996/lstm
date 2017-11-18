import tensorflow as tf
import numpy as np
from lstm_problems import BaseProblem, ABCSequenceProblem, AddSubNullProblem

summaries_dir = "/tmp/lstm_logs/"


class LSTMHandler:
    def __init__(self, h_length, act=tf.nn.tanh, **kwargs):
        problem = kwargs.get('problem')
        if problem is not None:
            assert isinstance(problem, BaseProblem)
            self.problem = problem
            x_length = problem.input_feature_size()
            y_length = problem.output_feature_size()
        else:
            x_length = kwargs.get('x_length')
            y_length = kwargs.get('y_length')

        assert x_length is not None and y_length is not None
        self.x_length = x_length
        self.h_length = h_length
        # Tensor dimensions are: [batch_size, sequence_length, x_length + h_length]
        self.data = tf.placeholder(tf.float32, [None, None, x_length])
        self.labels = tf.placeholder(tf.float32, [None, y_length])
        self.W_a = tf.Variable(tf.truncated_normal([h_length + x_length, h_length], -0.2, 0.1))
        self.b_a = tf.Variable(tf.zeros([1, h_length]))
        self.W_ifo = tf.Variable(tf.truncated_normal([x_length + h_length, 3 * h_length], -0.2, 0.1))
        self.b_ifo = tf.Variable(tf.zeros([1, 3 * h_length]))
        batch_size = tf.shape(self.data)[0]
        c_init = tf.zeros([batch_size, h_length])
        h_init = tf.zeros([batch_size, h_length])
        T = tf.shape(self.data)[1]
        t_less_T = lambda _, __, t: tf.less(t, T)
        h_T, c_T, _ = tf.while_loop(t_less_T, self.loop, (h_init, c_init, 0),
                                    parallel_iterations=32)
        print(h_T)
        self.W_pred = tf.Variable(tf.truncated_normal([h_length, y_length], -0.2, 0.1))
        self.b_pred = tf.Variable(tf.zeros([1, y_length]))
        self.prediction = tf.matmul(h_T, self.W_pred) + self.b_pred
        self.prob_pred = tf.nn.softmax(self.prediction)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                     logits=self.prediction)
        print(self.labels, self.prediction)
        self.mse = tf.losses.mean_squared_error(self.labels, self.prediction)
        self.trainer_mse = tf.train.AdamOptimizer().minimize(self.mse)
        self.trainer_cross_entropy = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

        self.mistakes = tf.not_equal(tf.argmax(self.prob_pred, 1), tf.argmax(self.labels, 1))
        self.rounded_pred = tf.round(tf.unstack(self.prediction, axis=1))
        self.mistakes_2 = tf.not_equal(tf.unstack(self.labels, axis=1), self.rounded_pred)
        self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))
        self.error_2 = tf.reduce_mean(tf.cast(self.mistakes_2, tf.float32))
        tf.summary.scalar('error', self.error)

    def unpack_data_to_equally_long_seqs(self, min_seq_len, max_seq_len, data):
        inp, out = data[0], data[1]
        assert len(inp) == len(out)
        out_dict = {}
        for length in range(min_seq_len, max_seq_len + 1):
            out_dict[length] = ([], [])
        for data_idx in range(len(inp)):
            data_point = inp[data_idx]
            if out_dict.get(len(data_point)) is not None:
                out_dict.get(len(data_point))[0].append(data_point)
                out_dict.get(len(data_point))[1].append(out[data_idx])
        return out_dict

    def train(self, **kwargs):
        problem = kwargs.get('problem')
        min_seq_len = kwargs.get('min_seq_len')
        max_seq_len = kwargs.get('max_seq_len')
        if problem is not None:
            assert isinstance(problem, BaseProblem)
            test_data = problem.test_data()
            train_data = problem.training_data()
            if min_seq_len is None:
                min_seq_len = problem.min_seq_length()
            if max_seq_len is None:
                max_seq_len = problem.max_seq_length()
        else:
            test_data = kwargs.get('test_data')
            train_data = kwargs.get('train_data')
        if self.problem is not None and test_data is None or train_data is None:
            test_data = self.problem.test_data()
            train_data = self.problem.training_data()
            if min_seq_len is None:
                min_seq_len = self.problem.min_seq_length()
            if max_seq_len is None:
                max_seq_len = self.problem.max_seq_length()
        print(np.shape(test_data), np.shape(train_data))
        assert test_data is not None and train_data is not None and min_seq_len is not None and max_seq_len is not None
        unpacked_test_data = self.unpack_data_to_equally_long_seqs(min_seq_len, 14, test_data)
        unpacked_train_data = self.unpack_data_to_equally_long_seqs(min_seq_len, 14, train_data)
        epoch = 250

        for i in range(epoch):
            if (i + 1) % 20 == 0:
                incorrect = 0
                for seq_length in range(min_seq_len, max_seq_len + 1):
                    summary, pred, lab, new_error = sess.run([merged, self.rounded_pred, self.labels, self.error_2],
                                                             {self.data: unpacked_test_data[seq_length][0],
                                                              self.labels: unpacked_test_data[seq_length][1]})
                    print(pred, lab)
                    test_writer.add_summary(summary, i)
                    incorrect += new_error
                incorrect /= (max_seq_len + 1 - min_seq_len)

                print('Epoch {:4d} | Test Accuracy {: 3.1f}%'.format(i + 1, incorrect * 100))
                incorrect = 0
                for seq_length in range(min_seq_len, max_seq_len + 1):
                    summary, new_error = sess.run([merged, self.error_2],
                                                  {self.data: unpacked_train_data[seq_length][0],
                                                   self.labels: unpacked_train_data[seq_length][1]})
                    train_writer.add_summary(summary, i)
                    incorrect += new_error
                incorrect /= (max_seq_len + 1 - min_seq_len)
                print('Epoch {:4d} | Train Accuracy  {: 3.1f}%'.format(i + 1, incorrect * 100))

            for seq_length in range(min_seq_len, max_seq_len + 1):
                batch_size = kwargs.get('batch_size')
                if batch_size is not None:
                    iter_len = int(np.shape(unpacked_train_data[seq_length])[1] / batch_size)
                    for j in range(iter_len + 1):
                        sess.run(self.trainer_mse, {self.data: unpacked_train_data[seq_length][0][j * batch_size:((j + 1) * batch_size) if j < iter_len else -1],
                                                    self.labels: unpacked_train_data[seq_length][1][j * batch_size:((j + 1) * batch_size) if j < iter_len else -1]})
                else:
                    sess.run(self.trainer_mse, {self.data: unpacked_train_data[seq_length][0],
                                                self.labels: unpacked_train_data[seq_length][1]})

    def lstm_step(self, xt, h_t_minus_1, c_t_minus_1, act=tf.nn.tanh):
        x_t_and_h_t_minus_1 = tf.concat([xt, h_t_minus_1], 1)
        # a_t is the usual activation but is called Candidate State as well,
        # because it is the unprocessed output you'd get in a standard RNN.
        a_t = act(tf.matmul(x_t_and_h_t_minus_1, self.W_a) + self.b_a)
        ifo_t = tf.sigmoid(tf.matmul(x_t_and_h_t_minus_1, self.W_ifo) + self.b_ifo)
        c_t = tf.add(tf.multiply(ifo_t[:, :self.h_length], a_t),
                     tf.multiply(ifo_t[:, self.h_length:2 * self.h_length], c_t_minus_1))
        h_t = tf.multiply(ifo_t[:, 2 * self.h_length:], act(c_t))
        return h_t, c_t

    def loop(self, h_t_minus_1, c_t_minus_1, t):
        h_t, c_t = self.lstm_step(self.data[:, t, :], h_t_minus_1, c_t_minus_1)
        return h_t, c_t, tf.add(t, 1)


if __name__ == '__main__':
    abc_problem = AddSubNullProblem()
    abc_problem.make_data(test_split=0.25)
    ls = LSTMHandler(80, problem=abc_problem)
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(summaries_dir + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + 'test', sess.graph)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ls.train(batch_size=10)

