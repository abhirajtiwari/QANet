import layers
import torch
import torch.nn as nn
import torch.nn.functional as F 


class QANET(nn.Module):
    """Implementation of QANET model for SQuAD 2.0
    """
    def __init__(self, word_vectors, hidden_size=128, drop_prob=0., context_len=1, question_len=1, word_embed=300):
        super(QANET, self).__init__()
        self.c_emb = layers.InputEmbeddingLayer(
            word_vectors=word_vectors,
            drop_prob=0.1
        )
        self.q_emb = layers.InputEmbeddingLayer(
            word_vectors=word_vectors,
            drop_prob=0.1
        )
        self.c_emb_enc = layers.EmbeddingEncoderLayer(
            conv_layers=4, 
            kernel_size=7,
            filters=128, 
            heads=8, 
            enc_blocks=1,
            drop_prob=drop_prob,
            sent_len=context_len, 
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
            sent_len=question_len,
            word_embed=word_embed,
            hidden_size=hidden_size
        )
        self.qc_att = layers.CQAttentionLayer(
            drop_prob=drop_prob
        )
        self.mod_enc = layers.ModelEncoderLayer(
            conv_layers=2,
            kernel_size=5,
            filters=128,
            heads=8,
            enc_blocks=7,
            drop_prob=drop_prob, 
            sent_len=context_len,
            word_embed=hidden_size, # new dim: [c, a, c.a, c.b]: qc_att.shape[1]
            hidden_size=hidden_size
        )
        self.start_out = layers.OutputLayer(
            drop_prob=drop_prob,
            word_embed=hidden_size # new dim: model_enc.shape[1]
        )
        self.end_out = layers.OutputLayer(
            drop_prob=drop_prob,
            word_embed=hidden_size  # new dim: model_enc.shape[1]
        )

    def forward(self, context, question, answer):
        # c_emb = self.c_emb(context)
        # q_emb = self.q_emb(question
        c_emb_enc = self.c_emb_enc(context)
        q_emb_enc  = self.q_emb_enc(question)
        qc_att = self.qc_att(c_emb_enc, q_emb_enc)
        mod_enc_1 = self.mod_enc(qc_att)
        mod_enc_2 = self.mod_enc(mod_enc_1)
        mod_enc_3 = self.mod_enc(mod_enc_2)
        start_out = self.start_out(mod_enc_1, mod_enc_2)
        end_out = self.end_out(mod_enc_1, mod_enc_3)
        score = start_out * end_out # some kind of probability distribution multiplication
        # return (start_out, end_out)
        return score


if __name__ == "__main__":
    torch.manual_seed(0)

    context = torch.randn((32, 300, 200))
    question = torch.randn((32, 300, 100))
    answer = torch.randn((32, 300, 150)) # part of context

    word_vectors = torch.randn((2, 3))

    qanet = QANET(word_vectors, hidden_size=128,
                  drop_prob=0., context_len=200, question_len=100, word_embed=300)
    # print(qanet(context, question, answer)[1].shape)
    print("Final score shape:")
    print(qanet(context, question, answer).shape)
