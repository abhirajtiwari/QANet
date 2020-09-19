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


class EmbeddingEncoderBlock(nn.Module):
    """
    """

    def __init__(self, conv_layers, kernel_size, filters, heads, drop_prob, sent_len, word_embed):
        super(EmbeddingEncoderBlock, self).__init__()
        self.pos_enc = nn.Embedding(sent_len, word_embed)

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.conv_0 = nn.Conv1d(
            word_embed, 128, kernel_size, padding=kernel_size//2)
        self.conv = nn.ModuleList([
            ConvBlock(word_embed=128, sent_len=sent_len,
                      in_channels=128, kernel_size=kernel_size)
            for _ in range(conv_layers - 1)
        ])

        # self.att = SelfAttention()
        # self.ff = nn.Linear()
        

    def forward(self, x):
        N, word_embed, sent_len = x.shape
        positions = torch.arange(0, sent_len).expand(N, sent_len)

        # Add positional Encoding:
        x = x.view(N, sent_len, word_embed) + self.pos_enc(positions)
        x = x.view(N, word_embed, sent_len)

        # Layer norm -> conv_0 (No residual block as 300!=128)
        x = self.layer_norm(x)
        x = self.conv_0(x)

        # Repetition of layer_norm and conv with residual block
        for layer in self.conv:
            x = layer(x)

        # Layer_norm -> Self Attention
        # Layer_norm -> Feed Forward
        return x
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


class EmbeddingEncoderLayer(nn.Module):
    """
    """

    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed):
        
        super(EmbeddingEncoderLayer, self).__init__()
        self.emb_enc = nn.ModuleList([
            EmbeddingEncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed)
            for _ in range(enc_blocks)
        ])

    def forward(self, x):
        for layer in self.emb_enc:
            x = layer(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn((32, 300, 100)) # (batch_size, word_embed, sent_len)
    
    x_l = EmbeddingEncoderLayer(conv_layers=4, kernel_size=7, filters=128,
                                heads=8, enc_blocks=1, drop_prob=0, sent_len=100, word_embed=300)(x)
    x_b = EmbeddingEncoderBlock(conv_layers=4, kernel_size=7, filters=128,
                                heads=8, drop_prob=0, sent_len=100, word_embed=300)(x)

    print(x.shape, x_l.shape, x_b.shape)
