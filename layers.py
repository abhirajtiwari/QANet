from torch import nn


# --------------- Model Layers ------------------

class InputEmbeddingLayer(nn.Module):
    """Input Embedding layer used by QANet

    Args:
        word_vectors (torch.Tensor): GLoVE vectors
    """
    def __init__(self, word_vectors):
        super(InputEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vectors)

    def forward(self, x):
        emb = self.embedding(x) # (batch_size, sequence_length, p1)

        return emb

class EmbeddingEncodeLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(EmbeddingEncoderLayer, self).__init__()

    def forward(self, x):

        return out


class CQAttentionLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(CQAttentionLayer, self).__init__()

    def forward(self, context, question):

        return out


class ModelEncoderLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(ModelEncoderLayer, self).__init__()

    def forward(self, x):

        return out


class OutputLayer(nn.Module):
    """Takes inputs from 2 of the ModelEncoderLayers

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(OutputLayer, self).__init__()

    def forward(self, a, b):

        return out


# ---------------- Helper Layers ----------------------        

class SelfAttention(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(SelfAttention, self).__init__()
        
    def forward(self, x):

        return out


class EncoderBlock(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, ):
        super(EmbeddingEncoderLayer, self).__init__()

    def forward(self, x):
        
        return out