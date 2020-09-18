import torch
import torch.nn as nn
import torch.nn.Functional as F


# --------------- Model Layers ------------------

class InputEmbeddingLayer(nn.Module):
    """
    Class that converts input words to their 300-d GLoVE word embeddings.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        """
        output shape: (batch_size, seq_len, embed_size=300)
        """
        super(InputEmbeddingLayer, self).__init__()
        self.word_embed_size = word_embed_size
        self.drop_prob = drop_prob 
        self.hidden_size = hidden_size

        self.embed = nn.Embedding.from_pretrained(word_vectors)
        

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)

        return emb

class EmbeddingEncoderLayer(nn.Module):
    """
    output shape: (batch_size, seq_len, enc_size=128)
    """

    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob):
        super(EmbeddingEncoderLayer, self).__init__()
        self.emb_enc = EncoderBlock(
            conv_layers=conv_layers, 
            kernel_size=kernel_size,
            filters=filters,
            heads=heads, 
            drop_prob=drop_prob)

    def forward(self, x):
        out = self.emb_enc(x)
        return out


class CQAttentionLayer(nn.Module):
    """
    """
    def __init__(self, drop_prob):
        super(CQAttentionLayer, self).__init__()

    def forward(self, context, question):
        pass
        # return out


class ModelEncoderLayer(nn.Module):
    """
    """

    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob):
        super(ModelEncoderLayer, self).__init__()
        self.model_enc = nn.ModuleList([
            EncoderBlock(
                conv_layers=conv_layers,
                kernel_size=kernel_size,
                filters=filters,
                heads=heads,
                drop_prob=drop_prob)
            for _ in range(enc_blocks)])

    def forward(self, x):
        out = self.model_enc(x)
        return out


class OutputLayer(nn.Module):
    """
    """
    def __init__(self, drop_prob):
        super(OutputLayer, self).__init__()

    def forward(self, a, b):
        pass
        # return out


# ---------------- Helper Layers ----------------------        

class SelfAttention(nn.Module):
    """
    """
    def __init__(self, heads, drop_prob):
        super(SelfAttention, self).__init__()
        
    def forward(self, x):
        pass
        # return out


class EncoderBlock(nn.Module):
    """
    """

    def __init__(self, conv_layers, kernel_size, filters, heads, drop_prob):
        super(EncoderBlock, self).__init__()

        self.conv = nn.ModuleList([nn.Conv2d(in_channels=_, out_channels=_, kernel_size=kernel_size)
                                   for _ in range(conv_layers)])
        self.att = SelfAttention(heads=heads, drop_prob=drop_prob)
        self.ff = nn.Linear()

    def forward(self, x):
        x = self.conv(x)
        x = self.att(x)
        out = self.ff(x)
        return out
