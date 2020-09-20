import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------- Model Layers ------------------
class InputEmbeddingLayer(nn.Module):
    def __init__(self, word_vectors, drop_prob):
        super(InputEmbeddingLayer, self).__init__()

        self.drop_prob = drop_prob 
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        
    def forward(self, x):
        emb = self.embed(x)   
        emb = F.dropout(emb, self.drop_prob, self.training)
        return emb  


class EmbeddingEncoderLayer(nn.Module):
    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed):
        super(EmbeddingEncoderLayer, self).__init__()

        self.emb_enc = nn.ModuleList([
            EmbeddingEncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=word_embed),
            *(EmbeddingEncoderBlock(conv_layers, kernel_size,
                                    filters, heads, drop_prob, sent_len, word_embed=128)
              for _ in range(enc_blocks-1))
        ])

    def forward(self, x):
        for layer in self.emb_enc:
            x = layer(x)
        return x


class CQAttentionLayer(nn.Module):
    def __init__(self, drop_prob):
        super(CQAttentionLayer, self).__init__()

    def forward(self, context, question):
        return out


class ModelEncoderLayer(nn.Module):
    """
    if enc_blocks > 1 -> positional encoding?
    """
    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed):
        super(ModelEncoderLayer, self).__init__()

        self.model_enc = nn.ModuleList([
            EmbeddingEncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=word_embed),
            *(EmbeddingEncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=128)
            for _ in range(enc_blocks-1))
        ])

    def forward(self, x):
        for layer in self.model_enc:
            x = layer(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, drop_prob):
        super(OutputLayer, self).__init__()

    def forward(self, a, b):
        return p


# ---------------- Helper Layers ----------------------
class SelfAttention(nn.Module):
    def __init__(self, word_embed, sent_len, heads, drop_prob):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        return x     


class EmbeddingEncoderBlock(nn.Module):
    def __init__(self, conv_layers, kernel_size, filters, heads, drop_prob, sent_len, word_embed):
        super(EmbeddingEncoderBlock, self).__init__()

        self.pos_enc = nn.Embedding(sent_len, word_embed)

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.conv_0 = nn.Conv1d(
            word_embed, 128, kernel_size, padding=kernel_size//2)

        self.conv = nn.ModuleList([
            ConvBlock(word_embed=word_embed, sent_len=sent_len,
                        out_channels=128, kernel_size=kernel_size),
            *(ConvBlock(word_embed=128, sent_len=sent_len,
                      out_channels=128, kernel_size=kernel_size)
            for _ in range(conv_layers - 1))
        ])
        self.att = SelfAttentionBlock(
            word_embed=128, sent_len=sent_len, heads=heads, drop_prob=drop_prob)
        self.ff = FeedForwardBlock(
            word_embed=128, sent_len=sent_len, in_features=128, out_features=128)

    def forward(self, x):
        N, word_embed, sent_len = x.shape
        positions = torch.arange(0, sent_len).expand(N, sent_len)

        # Add positional Encoding:
        x = x.permute(0, 2, 1) + self.pos_enc(positions)
        x = x.permute(0, 2, 1)

        # Layer norm -> conv_0 (No residual block as 300!=128)
        x = self.layer_norm(x)
        x = self.conv_0(x)

        # Repetition of layer_norm and conv with residual block
        for layer in self.conv:
            x = layer(x)

        # Layer_norm -> Self Attention
        x = self.att(x)
        # Layer_norm -> Feed Forward
        x = self.ff(x)
        return x


# ---------------- Helper Residual Blocks ----------------------
class ConvBlock(nn.Module):
    def __init__(self, word_embed, sent_len, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.word_embed = word_embed
        self.out_channels = out_channels

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.conv = nn.Conv1d(word_embed, out_channels,
                              kernel_size, padding=kernel_size//2)
        self.w_s = nn.Linear(word_embed, out_channels) # linear projection, read identity mapping(3.2 section) in ResNet paper

    def forward(self, x):
        x_l = self.layer_norm(x)
        if(self.word_embed != self.out_channels):
            x = self.w_s(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.conv(x_l)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, word_embed, sent_len, in_features, out_features):
        super(FeedForwardBlock, self).__init__()

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.ff = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        x_l = self.layer_norm(x)
        x = x + self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, word_embed, sent_len, heads, drop_prob):
        super(SelfAttentionBlock, self).__init__()

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.att = SelfAttention(word_embed=word_embed, sent_len=sent_len, heads=heads, drop_prob=drop_prob)

    def forward(self, x):
        x_l = self.layer_norm(x)
        x = x + self.att(x_l)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn((32, 300, 100)) # (batch_size, word_embed, sent_len)
    
    # x_b = EmbeddingEncoderBlock(conv_layers=4, kernel_size=7, filters=128,
    #                             heads=8, drop_prob=0, sent_len=100, word_embed=300)(x)
    # x_e = EmbeddingEncoderLayer(conv_layers=4, kernel_size=7, filters=128,
    #                             heads=8, enc_blocks=1, drop_prob=0, sent_len=100, word_embed=300)(x)
    # x_m = ModelEncoderLayer(conv_layers=2, kernel_size=5, filters=128,
    #                             heads=8, enc_blocks=1, drop_prob=0, sent_len=100, word_embed=300)(x)
    x_c = ConvBlock(word_embed=300, sent_len=100, out_channels=128, kernel_size=7)(x)
    # x_f = FeedForwardBlock(128, 100, 128, 128)(x_m)
    # x_a = SelfAttentionBlock(word_embed=128, sent_len=100, heads=8, drop_prob=0)(x_m)
    
    # print(x.shape, x_b.shape, x_e.shape, x_m.shape, x_c.shape, x_f.shape, x_a.shape, sep='\n')
    print(x_c.shape)