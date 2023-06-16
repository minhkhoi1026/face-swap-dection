import torch
from torch import nn

class FeatAttention(nn.Module):
    """
    Module to fusion two feature vector with add attention

    Args:
        feat_dim (int): dimension of the feature vector
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        # vector of 1s feat dim
        self.q = nn.Parameter(torch.ones(feat_dim))
        self.norm = nn.LayerNorm(feat_dim)
    
    def forward(self, X):
        # X (bs, 2, e) with bs=batch_size, e=embed_dim, normalize tokens to avoid scale
        X_norm = self.norm(X)
        # d (bs, 2,) multiple each feature vector with kernel
        d = torch.tensordot(X_norm, self.q, dims=1)
        # w (bs, 2) softmax vector as weight to fusion
        w = nn.functional.softmax(d, dim=1)
        
        # fusion two vector with w * x => (bs, e)
        return torch.sum(X * torch.unsqueeze(w, dim=-1), dim=1)

if __name__ == "__main__":
    feat_dim = 2
    bs = 4
    X = torch.randn(bs, 2, feat_dim)
    feat_attention = FeatAttention(feat_dim)
    print(feat_attention(X).shape)
