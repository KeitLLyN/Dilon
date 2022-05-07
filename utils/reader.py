import random
import numpy as np

from sklearn.model_selection import train_test_split

from utils.vocab import Vocabulary
import utils.service as service


class Reader:
    def __init__(self, data_path, vocab=Vocabulary()):
        self.vocab = vocab
        data = service.get_requests_from_file(data_path)

        data_to_length = list(map(self._data_to_length, data))
        self.data = [x[0] for x in data_to_length]
        self.lengths = [x[1] for x in data_to_length]

    def _data_to_length(self, req):
        seq = self.vocab.string_to_int(req)
        return seq, len(seq)


class Data(Reader):
    def __init__(self, data_path, vocab=Vocabulary(), predict=False):
        super(Data, self).__init__(data_path, vocab)

        if not predict:
            self._train_test_split()

    def _train_test_split(self):
        data, lengths = self._shuffle(self.data, self.lengths)
        x_train, self.x_test, y_train, self.y_test = train_test_split(data, lengths, test_size=0.1)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, test_size=0.2)

        self.test_size = len(self.x_test)

    def _shuffle(self, data, lengths):
        temp = list(zip(data, lengths))
        random.shuffle(temp)
        data, lengths = zip(*temp)

        return data, lengths

    def train_generator(self, batch_size, num_epochs):
        return service.batch_generator(
            self.x_train,
            self.y_train,
            num_epochs,
            batch_size,
            self.vocab)

    def val_generator(self):
        return service.one_by_one_generator(
            self.x_val,
            self.y_val)

    def test_generator(self):
        return service.one_by_one_generator(
            self.x_test,
            self.y_test)

    def predict_generator(self):
        return service.one_by_one_generator(
            self.data,
            self.lengths)
