import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------- Model Layers ------------------

class InputEmbeddingLayer(nn.Module):
    """
    Class that converts input words to their 300-d GLoVE word embeddings.
    """
    def __init__(self, word_vectors, drop_prob):
        """
        output shape: (batch_size, sent_len, word_embed_size=300)
        """
        super(InputEmbeddingLayer, self).__init__()
        self.drop_prob = drop_prob 
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, sent_len, word_embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)

        return emb  # (batch_size, sent_len, word_embed_size)

class EmbeddingEncoderLayer(nn.Module):
    """
    output shape: (batch_size, seq_len, enc_size=128)
    """

    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed_size):
        super(EmbeddingEncoderLayer, self).__init__()
        self.emb_enc = nn.ModuleList([
            EncoderBlock(
                conv_layers=conv_layers,
                kernel_size=kernel_size,
                filters=filters,
                heads=heads,
                drop_prob=drop_prob,
                sent_len=sent_len,
                word_embed_size=word_embed_size)
            for _ in range(enc_blocks-1)])

    def forward(self, x):
        out = self.emb_enc(x)
        return out


# class CQAttentionLayer(nn.Module):
#     """
#     """
#     def __init__(self, drop_prob):
#         super(CQAttentionLayer, self).__init__()

#     def forward(self, context, question):
#         pass
#         # return out


# class ModelEncoderLayer(nn.Module):
#     """
#     """

#     def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed_size):
#         super(ModelEncoderLayer, self).__init__()
#         self.positional_embedding = nn.Embedding(sent_len, word_embed_size)
#         self.model_enc = nn.ModuleList([
#             EncoderBlock(
#                 conv_layers=conv_layers,
#                 kernel_size=kernel_size,
#                 filters=filters,
#                 heads=heads,
#                 drop_prob=drop_prob,
#                 sent_len=sent_len,
#                 word_embed_size=word_embed_size)
#             for _ in range(enc_blocks-1)])

#     def forward(self, x):
#         x = self.positional_embedding(x)
#         out = self.model_enc(x)
#         return out


# class OutputLayer(nn.Module):
#     """
#     """
#     def __init__(self, drop_prob):
#         super(OutputLayer, self).__init__()

#     def forward(self, a, b):
#         pass
        # return out


# ---------------- Helper Layers ----------------------        

# class SelfAttention(nn.Module):
#     """
#     """
#     def __init__(self, heads, drop_prob):
#         super(SelfAttention, self).__init__()
        
#     def forward(self, x):
#         pass
#         # return out


class EncoderBlock(nn.Module):
    """
    """

    # def __init__(self, conv_layers, kernel_size, filters, heads, drop_prob, sent_len, embed_size):
    #     super(EncoderBlock, self).__init__()
    #     self.positional_embedding = nn.Embedding(sent_len, embed_size)
    #     self.conv = nn.ModuleList([
    #         ConvBlock(in_channels=_, out_channels=_, kernel_size=kernel_size)
    #                                for _ in range(conv_layers)])
    #     self.att = SelfAttention(heads=heads, drop_prob=drop_prob)
    #     self.ff = nn.Linear()
    #     self.layer_norm = nn.LayerNorm()

    # def forward(self, x):
    #     N, sent_len, word_embed_size = x.shape
    #     positions = torch.arange(0, sent_len).expand(N, sent_len)
    #     x = x + self.positional_embedding(positions)

    #     for layer in self.conv:
    #         x = layer()

        # return out
class ConvBlock(nn.Module):
    """
    """

    def __init__(self, word_embed, sent_len, in_channels, kernel_size):
        super(ConvBlock, self).__init__()

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        x_l = self.layer_norm(x)
        x_l = self.conv(x_l)
        out = x + x_l 
        return out



if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn((32, 300, 100)) # (batch_size, word_embed, sent_len)
    N, word_embed, sent_len = x.shape
    positions = torch.arange(0, sent_len).expand(N, sent_len)
    
    ### init
    
    pos_enc = nn.Embedding(sent_len, word_embed)
    layer_norm = nn.LayerNorm(x.shape[1:])
    conv_0 = nn.Conv1d(word_embed, 128, 7, padding=7//2)

    conv = nn.ModuleList([
        ConvBlock(128, sent_len, 128, 7),
        ConvBlock(128, sent_len, 128, 7),
        ConvBlock(128, sent_len, 128, 7)
    ])



    
    ### forward:

    # Conv_0: Embedding encoder layer: (only 1 block) i/p = 300, o/p = 128
    x = x.view(N, sent_len, word_embed) + pos_enc(positions) # adding positional embedding
    x = x.view(N, word_embed, sent_len) 
    x_l = layer_norm(x)
    x_l = conv_0(x_l)

    # Conv_1-3:
    for l in conv:
        x_r = l(x_l)

    
    print(x.shape, x_l.shape)
