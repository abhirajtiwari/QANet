import torch
from layers import *

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
    
    def __init__(self, word_vectors, d_model=300, c_len=100, q_len=100, hidden_state=128):
        super(QANet, self).__init__()
        self.c_emb = InputEmbeddingLayer(word_vectors)
        self.q_emb = InputEmbeddingLayer(word_vectors)
        self.c_enc = EmbeddingEncodeLayer(d_model, c_len, hidden_state)
        self.q_enc = EmbeddingEncodeLayer(d_model, q_len, hidden_state)
        self.cqa = CQAttentionLayer(hidden_state)
        self.model_enc = ModelEncoderLayer(hidden_state, c_len*4) 
        self.start_out = OutputLayer(hidden_state)
        self.end_out = OutputLayer(hidden_state)

    def forward(self, context, question):
        c_emb_enc = self.c_emb(context)  # (batch_size, hidden_size, c_len)
        q_emb_enc  = self.q_emb(question)  # (batch_size, hidden_size, q_len)

        qc_att = self.cqa(c_emb_enc, q_emb_enc)  # (batch_size, , c_len)

        mod_enc_1 = self.model_enc(qc_att)  # (batch_size, hidden_size, c_len)
        mod_enc_2 = self.model_enc(mod_enc_1)  # (batch_size, hidden_size, c_len)
        mod_enc_3 = self.model_enc(mod_enc_2)  # (batch_size, hidden_size, c_len)

        start_out = self.start_out(mod_enc_1, mod_enc_2)  # (batch_size, c_len)
        end_out = self.end_out(mod_enc_1, mod_enc_3)  # (batch_size, c_len)

        return start_out, end_out

if __name__ == "__main__":
    print("Hello, QANet")
