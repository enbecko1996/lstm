import abc
import numpy as np
import random


class BaseProblem(object):
    def __init__(self):
        self.data = None

    @abc.abstractmethod
    def input_feature_size(self):
        """Returns the amount of options for each position in the x sequence.
        For example sequences containing only the characters 'a', 'b' and 'c'
        have a feature size of 3."""

    @abc.abstractmethod
    def output_feature_size(self):
        """Returns the amount of options for the y_ vector.
        For example if the output can be either one of 'a', 'b' or 'c'
        It has a feature size of 3."""

    @abc.abstractmethod
    def make_data(self, test_split=0.25, **kwargs):
        """Initializing the training and test data if needed.
        For example loading from disk."""

    @abc.abstractmethod
    def training_data(self):
        """Returns a tuple containing the training data in the following format:
        (x, y_).
        x is a np array with the shape:
        [batch_size, sequence_length, feature_size]
        y_ is a np array with the shape:
        [batch_size, sequence_length]

        NOTE: the x tensors are encoded for the LSTM. It may not be human readable.
        Use decode_sequence(x[index]) to create a human readable sequence."""

    @abc.abstractmethod
    def test_data(self):
        """Returns a tuple containing the test data in the following format:
        (x, y_).
        x is a np array with the shape:
        [batch_size, sequence_length, feature_size]
        y_ is a np array with the shape:
        [batch_size, sequence_length]

        NOTE: the x tensors are encoded for the LSTM. It may not be human readable.
        Use decode_sequence(x[index]) to create a human readable sequence."""

    def decode_sequence(self, **kwargs):
        bin_seq = kwargs.get('bin_seq')
        if bin_seq is not None:
            s = ''
            for c in bin_seq:
                s += self.decode_character_bin(c)
            return s
        else:
            oh_seq = kwargs.get('oh_seq')
            if oh_seq is not None:
                s = ''
                for c in oh_seq:
                    s += self.decode_character_one_hot(c)
                return s

    @abc.abstractmethod
    def min_seq_length(self):
        pass

    @abc.abstractmethod
    def max_seq_length(self):
        pass

    def encode_character_bin(self, char):
        idx = self._CHARACTERS.index(char)
        binary = lambda n: n > 0 and [n & 1] + binary(n >> 1) or []
        bin_int = binary(idx)
        app = self._INPUT_FEATURE_SIZE - len(bin_int)
        out = [0] * app + bin_int[::-1]
        return out

    def encode_character_one_hot(self, char):
        idx = self._CHARACTERS.index(char)
        out = [0] * self._OUTPUT_FEATURE_SIZE
        out[idx] = 1
        return out

    def decode_character_bin(self, bin_rep):
        idx = 0
        for power in range(len(bin_rep)):
            idx += bin_rep[-(power + 1)] * (2 ** power)
        return self._CHARACTERS[idx]

    def decode_character_one_hot(self, oh_rep):
        idx = max(range(len(oh_rep)), key=lambda i: oh_rep[i])
        return self._CHARACTERS[idx]


class ABCSequenceProblem(BaseProblem):
    """LSTM-Problem asking for the next character in a sequence containing one repeating unit.
    For example the expectation for the sequence 'aadbcaadbcaad' would be a 'b' because the repeating
    unit is 'aadbc'. The repeating unit can have n characters, where n is 2, 3, 4 or 5. The sequence contains it
    twice plus the 0th to (n - 1)th character of the repeating unit.
    This results in sum_i=2^5(4^(i) * i) = 6432 possible sequences."""

    _INPUT_FEATURE_SIZE = 2
    _CHARACTERS = ['a', 'b', 'c', 'd']
    _OUTPUT_FEATURE_SIZE = len(_CHARACTERS)
    _MAX_UNIT_LEN = 5
    _POSSIBLE_SEQUENCES = 6368

    def __init__(self):
        super().__init__()

    def input_feature_size(self):
        return self._INPUT_FEATURE_SIZE

    def output_feature_size(self):
        return self._OUTPUT_FEATURE_SIZE

    def make_data(self, test_split=0.25, **kwargs):
        sequences = []
        for k in range(2):
            for c in self._CHARACTERS:
                for c2 in self._CHARACTERS:
                    seq = [self.encode_character_bin(c), self.encode_character_bin(c2)] * 2
                    for l in range(k + 1):
                        app = seq[l] if l < k else self.encode_character_one_hot(
                            self.decode_character_bin(seq[l]))
                        seq.append(app)
                    # print(self.decode_sequence(bin_seq=seq[:-1]), self.decode_character_one_hot(seq[-1]))
                    sequences.append(seq)
        for k in range(3):
            for c in self._CHARACTERS:
                for c2 in self._CHARACTERS:
                    for c3 in self._CHARACTERS:
                        seq = [self.encode_character_bin(c), self.encode_character_bin(c2), self.encode_character_bin(c3)] * 2
                        for l in range(k + 1):
                            app = seq[l] if l < k else self.encode_character_one_hot(
                                self.decode_character_bin(seq[l]))
                            seq.append(app)
                        # print(self.decode_sequence(bin_seq=seq[:-1]), self.decode_character_one_hot(seq[-1]))
                        sequences.append(seq)
        for k in range(4):
            for c in self._CHARACTERS:
                for c2 in self._CHARACTERS:
                    for c3 in self._CHARACTERS:
                        for c4 in self._CHARACTERS:
                            seq = [self.encode_character_bin(c), self.encode_character_bin(c2), self.encode_character_bin(c3),
                                   self.encode_character_bin(c4)] * 2
                            for l in range(k + 1):
                                app = seq[l] if l < k else self.encode_character_one_hot(
                                    self.decode_character_bin(seq[l]))
                                seq.append(app)
                            # print(self.decode_sequence(bin_seq=seq[:-1]), self.decode_character_one_hot(seq[-1]))
                            sequences.append(seq)
        for k in range(5):
            for c in self._CHARACTERS:
                for c2 in self._CHARACTERS:
                    for c3 in self._CHARACTERS:
                        for c4 in self._CHARACTERS:
                            for c5 in self._CHARACTERS:
                                seq = [self.encode_character_bin(c), self.encode_character_bin(c2), self.encode_character_bin(c3),
                                       self.encode_character_bin(c4), self.encode_character_bin(c5)] * 2
                                for l in range(k + 1):
                                    app = seq[l] if l < k else self.encode_character_one_hot(
                                        self.decode_character_bin(seq[l]))
                                    seq.append(app)
                                # print(self.decode_sequence(bin_seq=seq[:-1]), self.decode_character_one_hot(seq[-1]))
                                sequences.append(seq)
        random.shuffle(sequences)
        test_data = sequences[:int(test_split * len(sequences))]
        test_inp = [test_data[i][:-1] for i in range(len(test_data))]
        test_out = [test_data[i][-1] for i in range(len(test_data))]
        train_data = sequences[int(test_split * len(sequences)):]
        train_inp = [train_data[i][:-1] for i in range(len(train_data))]
        train_out = [train_data[i][-1] for i in range(len(train_data))]
        self.data = ((test_inp, test_out), (train_inp, train_out))
        print(np.shape(sequences), np.shape(test_inp), np.shape(test_out), np.shape(train_inp), np.shape(train_out))

    def training_data(self):
        return self.data[1]

    def test_data(self):
        return self.data[0]

    def min_seq_length(self):
        return 4

    def max_seq_length(self):
        return 14


class AddSubNullProblem(BaseProblem):
    _INPUT_FEATURE_SIZE = 2
    _CHARACTERS = ['+', '-', '0', 'I']
    _OUTPUT_FEATURE_SIZE = 1

    def __init__(self):
        super().__init__()

    def min_seq_length(self):
        return 6

    def max_seq_length(self):
        return 6

    def output_feature_size(self):
        return self._OUTPUT_FEATURE_SIZE

    def input_feature_size(self):
        return self._INPUT_FEATURE_SIZE

    def training_data(self):
        return self.data[1]

    def test_data(self):
        return self.data[0]

    def make_data(self, test_split=0.25, **kwargs):
        sequences = []
        for c in self._CHARACTERS:
            for c1 in self._CHARACTERS:
                for c2 in self._CHARACTERS:
                    for c3 in self._CHARACTERS:
                        for c4 in self._CHARACTERS:
                            for c5 in self._CHARACTERS:
                                seq = [self.encode_character_bin(c), self.encode_character_bin(c1),
                                       self.encode_character_bin(c2), self.encode_character_bin(c3),
                                       self.encode_character_bin(c4), self.encode_character_bin(c5)]
                                tmp = [c, c1, c2, c3, c4, c5]
                                invert = False
                                running_val = 0
                                for char in tmp:
                                    if (char == '+' and not invert) or (char == '-' and invert):
                                        running_val += 1
                                    elif (char == '-' and not invert) or (char == '+' and invert):
                                        running_val -= 1
                                    elif char == 'I':
                                        invert = not invert
                                print(self.decode_sequence(bin_seq=seq), running_val)
                                seq.append([running_val])
                                sequences.append(seq)
        random.shuffle(sequences)
        test_data = sequences[:int(test_split * len(sequences))]
        test_inp = [test_data[i][:-1] for i in range(len(test_data))]
        test_out = [test_data[i][-1] for i in range(len(test_data))]
        train_data = sequences[int(test_split * len(sequences)):]
        train_inp = [train_data[i][:-1] for i in range(len(train_data))]
        train_out = [train_data[i][-1] for i in range(len(train_data))]
        self.data = ((test_inp, test_out), (train_inp, train_out))
        print(np.shape(sequences), np.shape(test_inp), np.shape(test_out), np.shape(train_inp), np.shape(train_out))


if __name__ == '__main__':
    abcc = AddSubNullProblem()
    abcc.make_data(test_split=0.25)
