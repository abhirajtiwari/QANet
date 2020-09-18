from torch import nn

class InputEmbeddingLayer(nn.Module):
    """Input Embedding layer used by QANet

    Args:
        word_vectors (torch.Tensor): GLoVE vectors
        hidden_size (int): Size of hidden states (p1)
    """
    def __init__(self, word_vectors):
        super(InputEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vectors)

    def forward(self, x):
        emb = self.embedding(x) # (batch_size, sequence_length, p1)

        return emb
