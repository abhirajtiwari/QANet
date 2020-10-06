"""Top-level model classes.

Authors:
    Sahil Khose (sahilkhose18@gmail.com)
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
"""
from util import torch_from_json
import layers
import torch
import torch.nn as nn


class QANET(nn.Module):
    """QANet model for SQuAD 2.0

    Based on the paper:
    "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
    by Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, Quoc V. Le
    (https://arxiv.org/abs/1804.09541).

    Follows a high-level structure commonly found in SQuAD 2.0 models:
        - Input Embedding Layer: Embed word indices to get word vectors.
        - Embedding Encoder Layer: Encode the embedded sequence.
        - Context-Query Attention Layer: Apply a context-query attention mechanism to the encoded sequence.
        - Model Encoder Layer: Encode the sequence again.
        - Output Layer: Simple layer (e.g., fc + softmax) to get final outputs.   
    """
    def __init__(self, word_vectors, hidden_size=128, drop_prob=0., c_len=1, q_len=1, word_embed=300):
        """Init QANET Model.
        
        @param word_vectors (torch.Tensor): Pre-trained word vectors.
        @param hidden_size (int): Number of features in the hidden state at each layer.
        @param drop_prob (float): Dropout probability.
        @param c_len (int): Context sentence length. 
        @param q_len (int): Question sentence length. 
        @param word_embed (int): Pretrained word vector size. 
        """
        super(QANET, self).__init__()
        self.c_emb = layers.InputEmbeddingLayer(word_vectors=word_vectors, drop_prob=0.1)
        self.q_emb = layers.InputEmbeddingLayer(word_vectors=word_vectors, drop_prob=0.1)
        self.c_emb_enc = layers.EmbeddingEncoderLayer(
            conv_layers=4, 
            kernel_size=7,
            filters=128, 
            heads=8, 
            enc_blocks=1,
            drop_prob=drop_prob,
            sent_len=c_len, 
            word_embed=word_embed,
            hidden_size=hidden_size
        )
        self.q_emb_enc = layers.EmbeddingEncoderLayer(
            conv_layers=4,
            kernel_size=7,
            filters=128,
            heads=8,
            enc_blocks=1,
            drop_prob=drop_prob,
            sent_len=q_len,
            word_embed=word_embed,
            hidden_size=hidden_size
        )
        self.qc_att = layers.CQAttentionLayer(hidden_size=hidden_size, drop_prob=drop_prob)
        self.qc_conv = layers.ConvBlock(word_embed=hidden_size*4, sent_len=c_len, hidden_size=hidden_size, kernel_size=5)
        self.mod_enc = layers.ModelEncoderLayer(
            conv_layers=2,
            kernel_size=5,
            filters=128,
            heads=8,
            enc_blocks=7,
            drop_prob=drop_prob, 
            sent_len=c_len,
            word_embed=hidden_size,  
            hidden_size=hidden_size
        )
        self.start_out = layers.OutputLayer(drop_prob=drop_prob, word_embed=hidden_size) 
        self.end_out = layers.OutputLayer(drop_prob=drop_prob, word_embed=hidden_size)  

    def forward(self, context, question):
        """ Take a mini-batch of context and question sentences, compute the log-likelihood of each 
        position in the context being the start or end of an answer span. 
        

        @param context (List[List[str]]): List of context sentence tokens.
        @param question (List[List[str]]): List of question sentence tokens.
        @param answer (List[List[str]]): List of answer sentence tokens.

        @returns start_out (Tensor): Start probability distribution.
        @returns end_out (Tensor): End probability distribution.
        """
        c_emb = self.c_emb(context)  # (batch_size, word_embed, c_len)
        q_emb = self.q_emb(question)  # (batch_size, word_embed, q_len)

        c_emb_enc = self.c_emb_enc(c_emb)  # (batch_size, hidden_size, c_len)
        q_emb_enc  = self.q_emb_enc(q_emb)  # (batch_size, hidden_size, q_len)

        qc_att = self.qc_att(c_emb_enc, q_emb_enc)  # (batch_size, 4*hidden_size, c_len)
        qc_conv = self.qc_conv(qc_att)  # (batch_size, hidden_size, c_len)

        mod_enc_1 = self.mod_enc(qc_conv)  # (batch_size, hidden_size, c_len)
        mod_enc_2 = self.mod_enc(mod_enc_1)  # (batch_size, hidden_size, c_len)
        mod_enc_3 = self.mod_enc(mod_enc_2)  # (batch_size, hidden_size, c_len)


        start_out = self.start_out(mod_enc_1, mod_enc_2)  # (batch_size, c_len)
        end_out = self.end_out(mod_enc_1, mod_enc_3)  # (batch_size, c_len)

        return start_out, end_out


if __name__ == "__main__":
    torch.manual_seed(0)
    
    word_vec = torch_from_json("data/word_emb.json")
    context = torch.randn((2, 10, 20))
    question = torch.randn((2, 10, 10))
    # answer = torch.randn((32, 300, 150))  # part of context


    qanet = QANET(word_vec, hidden_size=8, drop_prob=0., c_len=20, q_len=10, word_embed=300)
    r = qanet(context, question)[0]
    
    print("Final score shape:")
    print(r.shape)
    print(r)
