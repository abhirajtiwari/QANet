"""Assortment of layers for use in models.py.

Authors:
    Sahil Khose (sahilkhose18@gmail.com)
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------- Model Layers ------------------
class InputEmbeddingLayer(nn.Module):
    """Embedding layer used in QANet, without character-level embedding.
    """
    def __init__(self, word_vectors, drop_prob):
        """
        @param word_vectors (torch.Tensor): Pre-trained word vectors.
        @param drop_prob (float): Probability of zero-ing out activations.
        """
        super(InputEmbeddingLayer, self).__init__()

        self.drop_prob = drop_prob 
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        
    def forward(self, x):
        """Looks up word embeddings for the words in a batch of sentences.
        @param x (torch.Tensor): (batch_size, sent_len)
        @returns emb (torch.Tensor): Word embeddings for the batch of sentences. (batch_size, word_embed, sent_len)
        """
        emb = self.embed(x)  # (batch_size, word_embed, sent_len)
        emb = F.dropout(emb, self.drop_prob, self.training)  # (batch_size, word_embed, sent_len)
        return emb  


class EmbeddingEncoderLayer(nn.Module):
    """Embedding Encoder layer which encodes using convolution, self attention and feed forward network.
    Takes input from Input Embedding Layer.
    """
    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed, hidden_size):
        """
        @param conv_layers (int): Number of convolution layers in one Encoder Block.
        @param kernel_size (int): Kernel size of the convolution layers.
        @param filters (int): Number of filters for the convolution layers. 
        @param heads (int): Number of heads for multihead attention. 
        @param enc_blocks (int): Number of Encoder Blocks. 
        @param drop_prob (float): Probability of zero-ing out activations.
        @param sent_len (int): Input sentence size. 
        @param word_embed (int): Pretrained word vector size. 
        @param hidden_size (int): Number of features in the hidden state at each layer.
        """
        super(EmbeddingEncoderLayer, self).__init__()

        self.emb_enc = nn.ModuleList([
            EncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=word_embed, hidden_size=hidden_size),
            *(EncoderBlock(conv_layers, kernel_size,
                                    filters, heads, drop_prob, sent_len, word_embed=hidden_size, hidden_size=hidden_size)
              for _ in range(enc_blocks-1))
        ])

    def forward(self, x):
        """Encodes the word embeddings.
        @param x (torch.Tensor): Word embeddings. (batch_size, word_embed, sent_len)
        @returns x (torch.Tensor): Encoded Word embeddings. (batch_size, hidden_size, sent_len)
        """
        for layer in self.emb_enc:
            x = layer(x)  # (batch_size, hidden_size, sent_len)
        return x


class CQAttentionLayer(nn.Module):
    """Context Query Attention Layer.
    Takes 2 inputs: Context Encoded Embedding and Question Encoded Embedding. 
    """
    def __init__(self, drop_prob):
        """
        @param drop_prob (float): Probability of zero-ing out activations.
        """
        super(CQAttentionLayer, self).__init__()

    def forward(self, context, question):
        """
        @param context (torch.Tensor): Encoded context embedding. (batch_size, hidden_size, c_len)
        @param question (torch.Tensor): Encoded question embedding. (batch_size, hidden_size, q_len)
        """
        return context  # (batch_size, hidden_size, c_len)


class ModelEncoderLayer(nn.Module):
    """Model Encoder layer which encodes using convolution, self attention and feed forward network.
    Takes input from CQAttention Layer.
    """
    def __init__(self, conv_layers, kernel_size, filters, heads, enc_blocks, drop_prob, sent_len, word_embed, hidden_size):
        """
        @param conv_layers (int): Number of convolution layers in one Encoder Block.
        @param kernel_size (int): Kernel size of the convolution layers.
        @param filters (int): Number of filters for the convolution layers. 
        @param heads (int): Number of heads for multihead attention. 
        @param enc_blocks (int): Number of Encoder Blocks. 
        @param drop_prob (float): Probability of zero-ing out activations.
        @param sent_len (int): Input sentence size. 
        @param word_embed (int): Word vector size. 
        @param hidden_size (int): Number of features in the hidden state at each layer.
        """
        super(ModelEncoderLayer, self).__init__()

        self.model_enc = nn.ModuleList([
            EncoderBlock(conv_layers, kernel_size,
                                  filters, heads, drop_prob, sent_len, word_embed=word_embed, hidden_size=hidden_size),
            *(EncoderBlock(conv_layers, kernel_size,
                           filters, heads, drop_prob, sent_len, word_embed=hidden_size, hidden_size=hidden_size)
            for _ in range(enc_blocks-1))
        ])

    def forward(self, x):
        """Encodes the word vectors.
        @param x (torch.Tensor): Input word vectors from CQAttention. (batch_size, , sent_len)
        @returns x (torch.Tensor): Encoded Word Vectors. (batch_size, hidden_size, sent_len)
        """
        for layer in self.model_enc:
            x = layer(x)  # (batch_size, hidden_size, sent_len)
        return x


class OutputLayer(nn.Module):
    """Output Layer which outputs the probability distribution for the answer span in the context span.
    Takes inputs from 2 Model Encoder Layers.
    """
    def __init__(self, drop_prob, word_embed):
        """
        @param drop_prob (float): Probability of zero-ing out activations.
        @param word_embed (int): Word vector size.
        """
        super(OutputLayer, self).__init__()
        self.ff = nn.Linear(2*word_embed, 1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, input_1, input_2):
        """Encodes the word embeddings.
        @param input_1 (torch.Tensor): Word vectors from first Model Encoder Layer. (batch_size, hidden_size, sent_len)
        @param input_2 (torch.Tensor): Word vectors from second Model Encoder Layer. (batch_size, hidden_size, sent_len)
        @returns p (torch.Tensor): Probability distribution for start/end token. (batch_size, sent_len)
        """
        x = torch.cat((input_1, input_2), dim=1)  # (batch_size, 2*hidden_size, sent_len)
        x = self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, 1, sent_len)
        p = self.soft(x)  # (batch_size, 1, sent_len)
        return p.squeeze()  # (batch_size, sent_len)


# ---------------- Helper Layers ----------------------
class EncoderBlock(nn.Module):
    """Encoder Block used in Input Embedding Layer and Model Embedding Layer. 
    """
    def __init__(self, conv_layers, kernel_size, filters, heads, drop_prob, sent_len, word_embed, hidden_size):
        """
        @param conv_layers (int): Number of convolution layers in one Encoder Block.
        @param kernel_size (int): Kernel size of the convolution layers.
        @param filters (int): Number of filters for the convolution layers. 
        @param heads (int): Number of heads for multihead attention.
        @param drop_prob (float): Probability of zero-ing out activations.
        @param sent_len (int): Input sentence size. 
        @param word_embed (int): Word vector size. 
        @param hidden_size (int): Number of features in the hidden state at each layer.
        """
        super(EncoderBlock, self).__init__()

        self.pos_enc = nn.Embedding(sent_len, word_embed)

        self.conv = nn.ModuleList([
            ConvBlock(word_embed=word_embed, sent_len=sent_len,
                        hidden_size=hidden_size, kernel_size=kernel_size),
            *(ConvBlock(word_embed=hidden_size, sent_len=sent_len,
                      hidden_size=hidden_size, kernel_size=kernel_size)
            for _ in range(conv_layers - 1))
        ])
        self.att = SelfAttentionBlock(
            hidden_size=hidden_size, sent_len=sent_len, heads=heads, drop_prob=drop_prob)
        self.ff = FeedForwardBlock(
            hidden_size=hidden_size, sent_len=sent_len)

    def forward(self, x):
        """Encodes the word vectors.
        @param x (torch.Tensor): Word vectors. (batch_size, word_embed, sent_len)
        @returns x (torch.Tensor): Encoded Word embeddings. (batch_size, hidden_size, sent_len)
        """
        N, word_embed, sent_len = x.shape  # N: batch_sizes
        positions = torch.arange(0, sent_len).expand(N, sent_len) 

        # Add positional Encoding:
        x = x.permute(0, 2, 1) + self.pos_enc(positions)  # (batch_size, word_embed, sent_len)
        x = x.permute(0, 2, 1)  # (batch_size, word_embed, sent_len)

        # Layer_norm -> conv 
        for layer in self.conv:
            x = layer(x)  # (batch_size, hidden_size, sent_len)

        # Layer_norm -> Self Attention
        x = self.att(x)  # (batch_size, hidden_size, sent_len)
        # Layer_norm -> Feed Forward
        x = self.ff(x)  # (batch_size, hidden_size, sent_len)
        return x


class SelfAttention(nn.Module):
    """Self Attention used in Self Attention Block for Encoder Block.
    """

    def __init__(self, hidden_size, sent_len, heads, drop_prob):
        """
        @param word_embed (int): Word vector size. 
        @param sent_len (int): Input sentence size. 
        @param heads (int): Number of heads for multihead attention. 
        @param drop_prob (float): Probability of zero-ing out activations.
        """
        super(SelfAttention, self).__init__()

    def forward(self, x):
        """
        @param x (torch.Tensor): Word vectors. (batch_size, hidden_size, sent_len)
        @returns x (torch.Tensor): Word vectors with self attention. (batch_size, hidden_size, sent_len)
        """
        return x


# ---------------- Helper Residual Blocks ----------------------
class ConvBlock(nn.Module):
    """Conv Block used in Encoder Block.
    """
    def __init__(self, word_embed, sent_len, hidden_size, kernel_size):
        """
        @param word_embed (int): Word vector size. 
        @param sent_len (int): Input sentence size. 
        @param out_channels (int): Number of output features.
        @param kernel_size (int): Kernel size of the convolution layers.
        """
        super(ConvBlock, self).__init__()
        self.word_embed = word_embed
        self.hidden_size = hidden_size

        self.layer_norm = nn.LayerNorm([word_embed, sent_len])
        self.conv = nn.Conv1d(word_embed, hidden_size, kernel_size, padding=kernel_size//2)
        self.w_s = nn.Linear(word_embed, hidden_size) # linear projection

    def forward(self, x):
        """
        @param x (torch.Tensor): Word vectors. (batch_size, word_embed, sent_len)
        @returns x (torch.Tensor): Word vectors. (batch_size, hidden_size, sent_len)

        Shortcut Connections based on the paper:
        "Deep Residual Learning for Image Recognition"
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        (https://arxiv.org/abs/1512.03385)

        For linear projection before Shortcut Connection:
            Refer Equation 2 (Section 3.2) in https://arxiv.org/pdf/1512.03385.pdf 

        """
        x_l = self.layer_norm(x)  # (batch_size, word_embed, sent_len)
        if(self.word_embed != self.hidden_size): 
            x = self.w_s(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, hidden_size, sent_len)
        x = x + self.conv(x_l)  # (batch_size, hidden_size, sent_len)
        return x


class SelfAttentionBlock(nn.Module):
    """Self Attention Block used in Encoder Block.
    """
    def __init__(self, hidden_size, sent_len, heads, drop_prob):
        """
        @param word_embed (int): Word vector size. 
        @param sent_len (int): Input sentence size. 
        @param heads (int): Number of heads for multihead attention.
        @param drop_prob (float): Probability of zero-ing out activations.
        """
        super(SelfAttentionBlock, self).__init__()

        self.layer_norm = nn.LayerNorm([hidden_size, sent_len])
        self.att = SelfAttention(hidden_size=hidden_size, sent_len=sent_len, heads=heads, drop_prob=drop_prob)

    def forward(self, x):
        """
        @param x (torch.Tensor): Word vectors. (batch_size, hidden_size, sent_len)
        @returns x (torch.Tensor): Word vectors with self attention. (batch_size, hidden_size, sent_len)
        """
        x_l = self.layer_norm(x)  # (batch_size, hidden_size, sent_len)
        x = x + self.att(x_l)  # (batch_size, hidden_size, sent_len)
        return x


class FeedForwardBlock(nn.Module):
    """Feed Forward Block used in Encoder Block.
    """

    def __init__(self, hidden_size, sent_len):
        """
        @param word_embed (int): Word vector size. 
        @param sent_len (int): Input sentence size.
        """
        super(FeedForwardBlock, self).__init__()

        self.layer_norm = nn.LayerNorm([hidden_size, sent_len])
        self.ff = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        @param x (torch.Tensor): Word vectors. (batch_size, hidden_size, sent_len)
        @returns x (torch.Tensor): Word vectors. (batch_size, hidden_size, sent_len)
        """
        x_l = self.layer_norm(x)  # (batch_size, hidden_size, sent_len)
        x = x + self.ff(x_l.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, hidden_size, sent_len)
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
