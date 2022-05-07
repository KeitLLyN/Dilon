from utils.vocab import Vocabulary


class Config:
    def __init__(self):
        self.batch_size = 128
        self.embed_size = 64
        self.hidden_size = 64
        self.num_layers = 2
        self.checkpoints = "checkpoints/"
        self.std_factor = 6.
        self.dropout = 0.7
        self.vocab = Vocabulary()
