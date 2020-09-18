import layers
import torch
import torch.nn as nn
import torch.nn.Functional as F 

class QANET(nn.module()):
    """Implementation of QANET model for SQuAD 2.0
    """
    def __init__(self, word_vectors, hidden_size=128, drop_prob=0.):
        super(QANET, self).__init__()
        self.emb = layers.InputEmbeddingLayer(
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            drop_prob=0.1
        )
        self.emb_enc = layers.EmbeddingEncoderLayer(
            conv_layers=4, 
            kernel_size=7,
            filters=128, 
            heads=8, 
            enc_blocks=1,
            drop_prob=drop_prob
        )
        self.qc_att = layers.CQAttentionLayer(
            drop_prob=drop_prob
        )
        self.mod_enc = layers.ModelEncoderLayer(
            conv_layers=2,
            kernel_size=5,
            filters=128,
            heads=8,
            enc_blocks=7,
            drop_prob=drop_prob
        )
        self.out = layers.OutputLayer(
            drop_prob=drop_prob
        )

    # def forward(self):

