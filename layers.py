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
    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed, hidden_size):
        super(EmbeddingEncoderLayer, self).__init__()

        self.emb_enc = nn.ModuleList([
            EncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=word_embed, hidden_size=hidden_size),
            *(EncoderBlock(conv_layers, kernel_size,
                                    filters, heads, drop_prob, sent_len, word_embed=hidden_size, hidden_size=hidden_size)
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
        return context


class ModelEncoderLayer(nn.Module):
    """
    """
    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed, hidden_size):
        super(ModelEncoderLayer, self).__init__()

        self.model_enc = nn.ModuleList([
            EncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=word_embed, hidden_size=hidden_size),
            *(EncoderBlock(conv_layers, kernel_size,
                           filters, heads, drop_prob, sent_len, word_embed=hidden_size, hidden_size=hidden_size)
            for _ in range(enc_blocks-1))
        ])

    def forward(self, x):
        for layer in self.model_enc:
            x = layer(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, drop_prob, word_embed):
        super(OutputLayer, self).__init__()
        self.ff = nn.Linear(2*word_embed, 1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, input_1, input_2):
        # (batch_size, word_embed, sent_len)
        x = torch.cat((input_1, input_2), dim=1) 
        x = self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.soft(x)
        return x


# ---------------- Helper Layers ----------------------
class SelfAttention(nn.Module):
    def __init__(self, word_embed, sent_len, heads, drop_prob):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        return x     


class EncoderBlock(nn.Module):
    def __init__(self, conv_layers, kernel_size, filters, heads, drop_prob, sent_len, word_embed, hidden_size):
        super(EncoderBlock, self).__init__()

        self.pos_enc = nn.Embedding(sent_len, word_embed)

        self.conv = nn.ModuleList([
            ConvBlock(word_embed=word_embed, sent_len=sent_len,
                        out_channels=hidden_size, kernel_size=kernel_size),
            *(ConvBlock(word_embed=hidden_size, sent_len=sent_len,
                      out_channels=hidden_size, kernel_size=kernel_size)
            for _ in range(conv_layers - 1))
        ])
        self.att = SelfAttentionBlock(
            word_embed=hidden_size, sent_len=sent_len, heads=heads, drop_prob=drop_prob)
        self.ff = FeedForwardBlock(
            word_embed=hidden_size, sent_len=sent_len, in_features=hidden_size, out_features=hidden_size)

    def forward(self, x):
        N, word_embed, sent_len = x.shape
        positions = torch.arange(0, sent_len).expand(N, sent_len)

        # Add positional Encoding:
        x = x.permute(0, 2, 1) + self.pos_enc(positions)
        x = x.permute(0, 2, 1)

        # Layer_norm -> conv 
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
    
    x_b = EncoderBlock(conv_layers=4, kernel_size=7, filters=128,
                                heads=8, drop_prob=0, sent_len=100, word_embed=300, hidden_size=128)(x)
    x_e = EmbeddingEncoderLayer(conv_layers=4, kernel_size=7, filters=128,
                                heads=8, enc_blocks=9, drop_prob=0, sent_len=100, word_embed=300, hidden_size=128)(x)
    x_m = ModelEncoderLayer(conv_layers=2, kernel_size=5, filters=128,
                            heads=8, enc_blocks=8, drop_prob=0, sent_len=100, word_embed=128, hidden_size=128)(x_e)
    
    print(x.shape, x_b.shape, x_e.shape, x_m.shape, sep='\n')
    print()

    smeb_1 = torch.randn((32, 200, 100))
    smeb_2 = torch.randn((32, 200, 100))

    out = OutputLayer(0., 200)
    print(out(smeb_1, smeb_2).shape)
