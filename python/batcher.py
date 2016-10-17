import numpy as np
import copy

class Batcher(object):
    def __init__(self, word2vec):
        self._premises = []
        self._hypothesis = []
        self._targets = []
        self._word2vec = word2vec
        self._embedding_dim = len(self._word2vec["beer"])
        self._out_of_voc_embedding = (2 * np.random.rand(self._embedding_dim) - 1) / 20
        self._delimiter = self._word2vec["_"]

    def batch_generator(self, dataset, num_epochs, batch_size, sequence_length):
        ids = range(len(dataset["targets"]))
        for epoch in range(num_epochs):
            permutation = np.random.permutation(ids)
            for i, idx in enumerate(permutation):
                self._premises.append(self.preprocess(sequence=dataset["premises"][idx], sequence_length=sequence_length))
                self._hypothesis.append(self.preprocess(sequence=dataset["hypothesis"][idx], sequence_length=sequence_length, is_delimiter=True))
                self._targets.append(dataset["targets"][idx])
                if len(self._targets) == batch_size or (i == (len(permutation) - 1) and epoch == (num_epochs - 1)):
                    batch = {
                                "premises": self._premises,
                                "hypothesis": self._hypothesis,
                                "targets": self._targets,
                            }
                    self._premises = []
                    self._hypothesis = []
                    self._targets = []
                    yield batch, epoch

    def preprocess(self, sequence, sequence_length, is_delimiter=False):
        p_seq = copy.deepcopy(sequence)
        preprocessed = []
        diff_size = len(p_seq) - sequence_length + int(is_delimiter)
        if diff_size  > 0:
            start_index = 0#np.random.randint(diff_size + 1)
            p_seq = p_seq[start_index: (start_index + sequence_length - int(is_delimiter))]
        for word in p_seq:
            try:
                embedding = self._word2vec[word]
            except KeyError:
                embedding = self._out_of_voc_embedding
            finally:
                preprocessed.append(embedding)
        if is_delimiter:
            preprocessed = [self._delimiter] + preprocessed
        for i in range(sequence_length - len(preprocessed)):
            preprocessed.append(np.zeros(self._embedding_dim))
        return preprocessed
