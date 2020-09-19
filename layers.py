import torch
from torch import nn
import torch.nn.functional as F


# --------------- Model Layers ------------------

class InputEmbeddingLayer(nn.Module):
    """Input Embedding layer used by QANet
    Word embedding of 300 D

    Args:
        word_vectors (torch.Tensor): GLoVE vectors (300-D)
        drop_prob (float): Dropout
    """
    def __init__(self, word_vectors, drop_prob=0.1):
        super(InputEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.dropout = torch.nn.Dropout(drop_prob)
    def forward(self, x):
        emb = self.embedding(x) # (batch_size, sequence_length, p1)
        emb = self.dropout(emb)
        return emb

class EmbeddingEncodeLayer(nn.Module):
    """Embedding Encoder Layer

    Args:
        d_model (int): Dimension of the word_vector
    """
    def __init__(self, d_model):
        super(EmbeddingEncoderLayer, self).__init__()
        self.enc_layer = EncoderBlock(d_model)
    def forward(self, x):
        return self.enc_layer(x)


class CQAttentionLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(CQAttentionLayer, self).__init__()

    def forward(self, context, question):

        return out


class ModelEncoderLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(ModelEncoderLayer, self).__init__()

    def forward(self, x):

        return out


class OutputLayer(nn.Module):
    """Takes inputs from 2 of the ModelEncoderLayers

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(OutputLayer, self).__init__()

    def forward(self, a, b):

        return out


# ---------------- Helper Layers ----------------------        

class SelfAttention(nn.Module): # FIXME This class is wrong
    """Multi-head self attention
    Refer to Attention is all you need paper to understand terminology

    Args:
        d_model (int): dimension of the word vector
        h (int): number of heads
    """
    def __init__(self, d_model, h=8):
        assert(d_model%h == 0)
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_v = self.d_model//h

        self.W_q = nn.Linear(in_features=self.d_v, out_features=self.d_v, bias=False)
        self.W_k = nn.Linear(in_features=self.d_v, out_features=self.d_v, bias=False)
        self.W_v = nn.Linear(in_features=self.d_v, out_features=self.d_v, bias=False)

        self.linear = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # Batch Size
        value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]

        # Split embedding in self.head pieces:
        values = values.reshape(N, value_len, self.h, self.d_v)
        keys = keys.reshape(N, key_len, self.h, self.d_v)
        queries = query.reshape(N, query_len, self.h, self.d_v)

        values = self.W_v(values)
        keys = self.W_k(keys)
        queries = self.W_q(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.d_model ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.h*self.d_v)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum: (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.linear(out)
        return out

class EncoderBlock(nn.Module):
    """Encoder Block for QANet
    convolution-layer × # + self-attention-layer + feed-forward-layer

    Args:
        nn ([type]): [description]
    """
    def __init__(self, d_model, conv_layer=4, kernel_size=7, filters=128, heads=8):
        super(EmbeddingEncoderLayer, self).__init__()

        # TODO positional encoding sometime

        self.conv = nn.ModuleList(
            [ConvolutionBlock(c_in=d_model,c_out=filters, kernel_size=kernel_size) for _ in range(conv_layer)] # FIXME is c_in right??
        )
        self.attn = SelfAttentionBlock(d_model)
        self.ff = FeedForwardBlock(filters, filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        x = self.ff(x)
        return x

# -------------------- Residual Blocks ----------------------

class ConvolutionBlock(nn.Module):
    def __init__(self, c_in, sent_len, kernel_size):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size, bias=False, padding=(kernel_size//2))
        self.layer_norm = nn.LayerNorm([c_in, sent_len]) 
    def forward(self, x):
        return x + self.conv(self.layer_norm(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, sent_len):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = SelfAttention(d_model)
        self.layer_norm = nn.LayerNorm([d_model, sent_len]) 
    def forward(self, x):
        a = self.layer_norm(x)
        att = self.self_attn(a, a, a)
        att = att.permute(0, 2, 1)

        return x + att

class FeedForwardBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForwardBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(in_features) # FIXME add the layer norm parameters, what is normalised_shape
    def forward(self, x):
        return x + self.ff(self.layer_norm(x))


if __name__ == "__main__":
    a = torch.randn((2, 128, 32))
    conv_block = ConvolutionBlock(128, 32, 7)
    attn_block = SelfAttentionBlock(128, 32)
    print(conv_block(a).size(), attn_block(a).size())