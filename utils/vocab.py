import os
import numpy as np
import json


class Vocabulary:
    def __init__(self):
        self.vocab_file = 'utils/vocab.json'
        with open(self.vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def string_to_int(self, text):
        # text = list(text)
        # text.append('<EOS>')
        # [self.vocab.get(char, self.vocab['<UNK>']) for char in chars]
        res = []
        # for requets in text:
        for char in text:
            number = self.vocab.get(char, self.vocab['<UNK>'])
            res.append(number)
        return res

    def int_to_string(self, char_ids):
        return [self.reverse_vocab[i] for i in char_ids]
