"""Top-level model classes.

Authors:
    Sahil Khose (sahilkhose18@gmail.com)
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
"""
from util import torch_from_json
import layers
import torch
import torch.nn as nn


class QANet(nn.Module):
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
    def __init__(self, word_vectors, hidden_size=64, drop_prob=0., c_len=400, q_len=50, word_embed=300, heads=4):
        """Init QANET Model.
        
        @param word_vectors (torch.Tensor): Pre-trained word vectors.
        @param hidden_size (int): Number of features in the hidden state at each layer.
        @param drop_prob (float): Dropout probability.
        @param c_len (int): Context sentence length. 
        @param q_len (int): Question sentence length. 
        @param word_embed (int): Pretrained word vector size. 
        """
        super(QANet, self).__init__()
        self.c_emb = layers.InputEmbeddingLayer(word_vectors=word_vectors, drop_prob=0.1)
        self.q_emb = layers.InputEmbeddingLayer(word_vectors=word_vectors, drop_prob=0.1)
        self.c_emb_enc = layers.EmbeddingEncoderLayer(
            conv_layers=3, 
            kernel_size=7,
            filters=64, 
            heads=heads, 
            enc_blocks=1,
            drop_prob=drop_prob,
            sent_len=c_len, 
            word_embed=word_embed,
            hidden_size=hidden_size
        )
        self.q_emb_enc = layers.EmbeddingEncoderLayer(
            conv_layers=3,
            kernel_size=7,
            filters=64,
            heads=heads,
            enc_blocks=1,
            drop_prob=drop_prob,
            sent_len=q_len,
            word_embed=word_embed,
            hidden_size=hidden_size
        )
        self.qc_att = layers.CQAttentionLayer(hidden_size=hidden_size, drop_prob=drop_prob)
        self.qc_conv = layers.ConvBlock(word_embed=hidden_size*4, sent_len=c_len, hidden_size=hidden_size, kernel_size=5)
        self.mod_enc = layers.ModelEncoderLayer(
            conv_layers=3,
            kernel_size=5,
            filters=64, 
            heads=heads,
            enc_blocks=4,
            drop_prob=drop_prob,
            sent_len=c_len, 
            word_embed=hidden_size, 
            hidden_size=hidden_size
        )
        self.start_out = layers.OutputLayer(drop_prob=drop_prob, word_embed=hidden_size) 
        self.end_out = layers.OutputLayer(drop_prob=drop_prob, word_embed=hidden_size)  

    def forward(self, cw_idxs, qw_idxs):
        """ Take a mini-batch of context and question sentences, compute the log-likelihood of each 
        position in the context being the start or end of an answer span. 
        

        @param context (List[List[str]]): List of context sentence tokens.
        @param question (List[List[str]]): List of question sentence tokens.
        @param answer (List[List[str]]): List of answer sentence tokens.

        @returns start_out (Tensor): Start probability distribution.
        @returns end_out (Tensor): End probability distribution.
        """
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs  # (batch_size, c_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs  # (batch_size, q_len)   
        # c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        # print("__"*80)
        # print("__"*80)
        # print("C_EMB, Q_EMB")
        c_emb = self.c_emb(cw_idxs)  # (batch_size, word_embed, c_len)
        q_emb = self.q_emb(qw_idxs)  # (batch_size, word_embed, q_len)
        # print("__"*80)
        # print("C_EMB_ENC, Q_EMB_ENC")
        # print("c_emb shape: ", c_emb.shape)
        # print("c mask shape: ", c_mask.shape)
        c_emb_enc = self.c_emb_enc(c_emb, c_mask)  # (batch_size, hidden_size, c_len)
        q_emb_enc  = self.q_emb_enc(q_emb, q_mask)  # (batch_size, hidden_size, q_len)
        # print("__"*80)
        # print("CQAttention")
        qc_att = self.qc_att(c_emb_enc, q_emb_enc, c_mask, q_mask)  # (batch_size, 4*hidden_size, c_len) # ! Add c_mask, q_mask here
        qc_conv = self.qc_conv(qc_att)  # (batch_size, hidden_size, c_len)
        # print("__"*80)
        # print("MOD_ENC")
        mod_enc_1 = self.mod_enc(qc_conv, c_mask)  # (batch_size, hidden_size, c_len)
        mod_enc_2 = self.mod_enc(mod_enc_1, c_mask)  # (batch_size, hidden_size, c_len)
        mod_enc_3 = self.mod_enc(mod_enc_2, c_mask)  # (batch_size, hidden_size, c_len)
        # print("__"*80)
        # print("OUTPUT")
        start_out = self.start_out(mod_enc_1, mod_enc_2, c_mask)  # (batch_size, c_len)
        end_out = self.end_out(mod_enc_1, mod_enc_3, c_mask)  # (batch_size, c_len)
        # print("__"*80)
        # print("__"*80)
        return start_out, end_out

if __name__ == "__main__":
    torch.manual_seed(0)
    
    word_vec = torch_from_json("./data/word_emb.json")
    # word_vec = torch.randn(2, 3)
    context = torch.rand((2, 200)).to(torch.int64)
    question = torch.rand((2, 100)).to(torch.int64)
    # answer = torch.randn((32, 300, 150))  # part of context


    qanet = QANet(word_vec, hidden_size=8, drop_prob=0., c_len=200, q_len=100, word_embed=300, heads=8)
    r = qanet(context, question)[0]
    
    print("Final score shape:")
    print(r.shape)  # (batch_size, sent_len) (2, 20)
    # print(r)


    #################################################################################
    ### decoder masks from transformers checking
    #################################################################################
    # trg = torch.tensor(
    #     [
    #         [3, 9, 5, 2, 0],
    #         [4, 5, 6, 0, 0]
    #     ]
    # ).float()
    # N, trg_len = trg.shape
    # trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
    # print(trg_mask)
    #################################################################################
    # tensor([[[[1., 0., 0., 0., 0.],
    #           [1., 1., 0., 0., 0.],
    #           [1., 1., 1., 0., 0.],
    #           [1., 1., 1., 1., 0.],
    #           [1., 1., 1., 1., 1.]]],


    #         [[[1., 0., 0., 0., 0.],
    #           [1., 1., 0., 0., 0.],
    #             [1., 1., 1., 0., 0.],
    #             [1., 1., 1., 1., 0.],
    #             [1., 1., 1., 1., 1.]]]])
    #################################################################################
    # trg = trg.unsqueeze(1)
    # print(trg.shape, trg_mask.shape)
    # layer = layers.SelfAttention(hidden_size=1, heads=1, drop_prob=0.)

    # print(layer(trg, trg, trg, trg_mask))  # print energy, attention, out
    #################################################################################
    ## energy:
    # tensor([[[[-3.6146e-02, -1.0000e+20, -1.0000e+20, -1.0000e+20, -1.0000e+20],
    #           [-1.0844e-01, -3.2532e-01, -1.0000e+20, -1.0000e+20, -1.0000e+20],
    #           [-6.0244e-02, -1.8073e-01, -1.0041e-01, -1.0000e+20, -1.0000e+20],
    #           [-2.4098e-02, -7.2293e-02, -4.0163e-02, -1.6065e-02, -1.0000e+20],
    #           [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]],


    #         [[[-6.4260e-02, -1.0000e+20, -1.0000e+20, -1.0000e+20, -1.0000e+20],
    #           [-8.0325e-02, -1.0041e-01, -1.0000e+20, -1.0000e+20, -1.0000e+20],
    #             [-9.6390e-02, -1.2049e-01, -1.4459e-01, -1.0000e+20, -1.0000e+20],
    #             [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+20],
    #             [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]]],
    #        grad_fn= < MaskedFillBackward0 > )

    ## attention:
    # tensor([[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #           [0.5540, 0.4460, 0.0000, 0.0000, 0.0000],
    #           [0.3512, 0.3114, 0.3374, 0.0000, 0.0000],
    #           [0.2535, 0.2416, 0.2494, 0.2555, 0.0000],
    #           [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]]],


    #         [[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #           [0.5050, 0.4950, 0.0000, 0.0000, 0.0000],
    #             [0.3414, 0.3333, 0.3253, 0.0000, 0.0000],
    #             [0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
    #             [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]]]],
    #        grad_fn= < SoftmaxBackward > )

    ## out:
    # tensor([[[-2.4691],
    #          [-4.6716],
    #          [-4.5621],
    #          [-3.8623],
    #          [-3.1276]],

    #         [[-3.2922],
    #          [-3.6996],
    #          [-4.1020],
    #          [-3.0864],
    #          [-2.4691]]], grad_fn= < ViewBackward > )

    ## out: after linear
    # tensor([[[1.8171],
    #      [3.4380],
    #      [3.3574],
    #      [2.8424],
    #      [2.3017]],

    #     [[2.4228],
    #      [2.7227],
    #      [3.0188],
    #      [2.2714],
    #      [1.8171]]], grad_fn= < UnsafeViewBackward >)
    #################################################################################



    #################################################################################
    ### self attention masks
    #################################################################################

    # cw_idxs = torch.tensor(
    #     [
    #         [3, 9, 5, 2, 0],
    #         [4, 5, 6, 0, 0]
    #     ]
    # ).float()
    # c_mask = torch.zeros_like(cw_idxs) != cw_idxs
    # cw_idxs = cw_idxs.unsqueeze(1) 
    # print(cw_idxs.shape, c_mask.shape)  # (2, 1, 5), (2, 5)
    # layer = layers.CQAttentionLayer(hidden_size=1, drop_prob=0.)
    # print(layer(cw_idxs, cw_idxs, c_mask, c_mask))


    # layer = layers.SelfAttention(hidden_size=1, heads=1, drop_prob=0.)


    # print(layer(cw_idxs, cw_idxs, cw_idxs, c_mask))  # print energy, attention, out

    # ## energy:

    # tensor([[[[-3.6146e-02, -1.0844e-01, -6.0244e-02, -2.4098e-02, -1.0000e+20],
    #           [-1.0844e-01, -3.2532e-01, -1.8073e-01, -7.2293e-02, -1.0000e+20],
    #           [-6.0244e-02, -1.8073e-01, -1.0041e-01, -4.0163e-02, -1.0000e+20],
    #           [-2.4098e-02, -7.2293e-02, -4.0163e-02, -1.6065e-02, -1.0000e+20],
    #           [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+20]]],


    #         [[[-6.4260e-02, -8.0325e-02, -9.6390e-02, -1.0000e+20, -1.0000e+20],
    #           [-8.0325e-02, -1.0041e-01, -1.2049e-01, -1.0000e+20, -1.0000e+20],
    #             [-9.6390e-02, -1.2049e-01, -1.4459e-01, -1.0000e+20, -1.0000e+20],
    #             [0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+20, -1.0000e+20],
    #             [0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+20, -1.0000e+20]]]],
    #        grad_fn= < MaskedFillBackward0 > )

    # ## attention: energy softmax

    # tensor([[[[0.2552, 0.2374, 0.2491, 0.2583, 0.0000],
    #           [0.2651, 0.2134, 0.2466, 0.2749, 0.0000],
    #           [0.2586, 0.2292, 0.2484, 0.2638, 0.0000],
    #           [0.2535, 0.2416, 0.2494, 0.2555, 0.0000],
    #           [0.2500, 0.2500, 0.2500, 0.2500, 0.0000]]],


    #         [[[0.3387, 0.3333, 0.3280, 0.0000, 0.0000],
    #           [0.3400, 0.3333, 0.3267, 0.0000, 0.0000],
    #             [0.3414, 0.3333, 0.3253, 0.0000, 0.0000],
    #             [0.3333, 0.3333, 0.3333, 0.0000, 0.0000],
    #             [0.3333, 0.3333, 0.3333, 0.0000, 0.0000]]]],
    #        grad_fn= < SoftmaxBackward > )

    # ## out: attention * values

    # tensor([[[-3.8390],
    #          [-3.7028],
    #          [-3.7928],
    #          [-3.8623],
    #          [-3.9095]],

    #         [[-4.1064],
    #          [-4.1042],
    #          [-4.1020],
    #          [-4.1152],
    #          [-4.1152]]], grad_fn= < ViewBackward > )

    ## out: linear(out)
    # tensor([[[2.8252],
    #          [2.7250],
    #          [2.7913],
    #          [2.8424],
    #          [2.8771]],

    #         [[3.0221],
    #          [3.0204],
    #          [3.0188],
    #          [3.0286],
    #          [3.0286]]], grad_fn= < UnsafeViewBackward > )
