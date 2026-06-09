import torch
import torch.nn as nn


class BaselineHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if mlp_dim and mlp_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mlp_dim, 1),
            )
        else:
            self.net = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: [B, T, H]
        return self.net(self.dropout(h)).squeeze(-1)  # [B, T]
