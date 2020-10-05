import torch
from torch import nn
import torch.nn.functional as F
import math


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
        return emb.permute(0,2,1)

class EmbeddingEncodeLayer(nn.Module):
    """Embedding Encoder Layer

    Args:
        d_model (int): Dimension of the word_vector
        sent_len (int): Length of the sentence
    """
    def __init__(self, d_model, sent_len, hidden_state, heads=8):
        super(EmbeddingEncodeLayer, self).__init__()
        self.enc_layer = EncoderBlock(d_model, sent_len, filters=hidden_state, heads=heads)
    def forward(self, x):
        return self.enc_layer(x)


class CQAttentionLayer(nn.Module):
    """context query attention layer 
    understood and influenced from https://github.com/tailerr/QANet-pytorch/

    Args:
        context (torch.Tensor): Encoded context vectors
        queries (torch.Tensor): Encoded queries vectors
    """
    def __init__(self, d_model, drop_prob=0.1):
        super(CQAttentionLayer, self).__init__()
        self.d_model = d_model

        w = torch.empty(d_model*3)
        lim = 1/d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)
        self.dropout = drop_prob

    def forward(self, context, query):
        context = context.permute(0,2,1)
        query = query.permute(0,2,1)
        c_len = context.size(1)
        q_len = query.size(1)
        # nn.functional.dropout(context, self.dropout, self.training, True)
        # nn.functional.dropout(query, self.dropout, self.training, True)
        c = context.repeat(q_len, 1, 1, 1).permute([1, 0, 2, 3])
        q = query.repeat(c_len, 1, 1, 1).permute([1, 2, 0, 3])
        cq = c*q
        s = torch.matmul(torch.cat((q, c, cq), 3), self.w).transpose(1, 2)
        s1 = nn.functional.softmax(s, 1)
        s2 = nn.functional.softmax(s, 1)
        
        a = torch.bmm(s1, query)
        l = torch.bmm(s1, s2.transpose(1, 2))
        b = torch.bmm(l, context)
        output = torch.cat((context, a, context*a, context*b), dim=2) # XXX concat over hidden_state only
        return nn.functional.dropout(output, p=self.dropout).permute(0,2,1)


class ModelEncoderLayer(nn.Module):
    """Model Encoder Block

    Args:
        d_model (int): Dimenstion of the input vector
        sent_len (int): Length of the input sentence
    """
    def __init__(self, d_model, sent_len, enc_blocks=4, conv_layer=2, heads=8):
        super(ModelEncoderLayer, self).__init__()

        self.model_enc = nn.ModuleList([
            EncoderBlock(d_model, sent_len, conv_layer=conv_layer, heads=heads),
            *(EncoderBlock(d_model, sent_len, conv_layer=conv_layer, heads=heads)
            for _ in range(enc_blocks-1))
        ])

    def forward(self, x):
        for layer in self.model_enc:
            x = layer(x)
        return x


class OutputLayer(nn.Module):
    """Takes inputs from 2 of the ModelEncoderLayers

    Args:
        d_model (int): Dimension of the input vector (should be 128 according to qanet)
    """
    def __init__(self, d_model):
        super(OutputLayer, self).__init__()
        self.lin = nn.Linear(in_features=2*d_model,out_features=1)
        self.s = nn.Softmax(dim=2)

    def forward(self, in1, in2):
        x = torch.cat((in1, in2), dim=1)
        x = self.lin(x.permute(0,2,1)).permute(0, 2, 1)
        x = self.s(x)
        return x.squeeze()


# ---------------- Helper Layers ----------------------        

class SelfAttention(nn.Module): 
    """Multi-head self attention
    Refer to Attention is all you need paper to understand terminology

    Args:
        d_model (int): dimension of the word vector
        h (int): number of heads
    """
    def __init__(self, d_model, h=8):
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
        d_model (int): Dimension of the input vectors
        sent_len (int): Length of a sentence
        conv_layer (int): Number of times the conv block is repeated 
        kernel_size (int): Size of the kernel for 1d convolutions
        filters (int): Number of filters
        heads (int): Number of heads in multihead attention
    """
    def __init__(self, d_model, sent_len, conv_layer=3, kernel_size=7, filters=128, heads=8):
        super(EncoderBlock, self).__init__()

        self.pos_enc = PositionalEncoder(sent_len, d_model)
        self.conv1 = ConvolutionBlock(d_model, filters, sent_len, kernel_size)
        self.conv = nn.ModuleList(
            [ConvolutionBlock(c_in=filters, c_out=filters ,sent_len=sent_len, kernel_size=kernel_size) for _ in range(conv_layer-1)]
        )
        self.attn = SelfAttentionBlock(filters, sent_len, heads)
        self.ff = FeedForwardBlock(filters, filters, sent_len)

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.conv1(x)
        for layer in self.conv:
            x = layer(x)
        x = self.attn(x)
        x = self.ff(x)
        return x

class PositionalEncoder(nn.Module):
    """Generate positional encoding for a vector

    Args:
        length (int): length of the input sentence to be encoded
        d_model (int): dimention of the word vector

    Returns:
        torch.Tensor: positionaly encoded vector
    """
    def __init__(self, length, d_model):
        super(PositionalEncoder, self).__init__()

        f = torch.Tensor([10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]).unsqueeze(dim=1)
        phase = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, f), phase)), requires_grad=False)

    def forward(self, x):
        return x + self.pos_encoding[0:x.size(1)]

# -------------------- Residual Blocks ----------------------

class ConvolutionBlock(nn.Module):
    def __init__(self, c_in, c_out, sent_len, kernel_size):
        super(ConvolutionBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, bias=False, padding=(kernel_size//2))
        self.layer_norm = nn.LayerNorm([c_in, sent_len]) 
        self.w_s = nn.Linear(c_in, c_out)
    def forward(self, x):
        ln = self.layer_norm(x)
        if (self.c_in != self.c_out):
            x = self.w_s(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + self.conv(ln)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, sent_len, heads):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = SelfAttention(d_model,heads)
        self.layer_norm = nn.LayerNorm([d_model, sent_len]) 
    def forward(self, x):
        a = self.layer_norm(x)
        att = self.self_attn(a, a, a)
        att = att.permute(0, 2, 1)

        return x + att

class FeedForwardBlock(nn.Module):
    def __init__(self, in_features, out_features, sent_len):
        super(FeedForwardBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm([in_features, sent_len]) 
    def forward(self, x):
        ln = self.layer_norm(x)
        return x + self.ff(ln.permute(0, 2, 1)).permute(0, 2, 1)


if __name__ == "__main__":
    c = torch.randn((2, 300, 32))
    q = torch.randn((2, 300, 16))

    enc = EmbeddingEncodeLayer(300, 32, 128)
    encq = EmbeddingEncodeLayer(300, 16, 128)
    cqa = CQAttentionLayer(128, 0.5)
    cq_conv = ConvolutionBlock(4*128, 128, 32, 5)
    modenc = ModelEncoderLayer(128, 32)
    start = OutputLayer(128)
    end = OutputLayer(128)

    ans = cqa(enc(c),encq(q))
    ans = cq_conv(ans) # XXX this makes the dimensions right
    ans1 = modenc(ans)
    ans2 = modenc(ans1)
    ans3 = modenc(ans2)
    
    s_p = start(ans1, ans2)
    e_p = end(ans1, ans3)

    print (s_p.size(), e_p.size())
    print("All the functions are dimentionally correct!")
    print("Only the input embedding class is not checked by this script.")