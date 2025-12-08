import torch
import torch.nn as nn

# TODO: stack attention heads like so
# class GlucoseTransformerEncoder(nn.Module):
    # class TransformerBlock(nn.Module):
    #     def __init__(self, embedding_dim, num_heads, ff_dim):
    #         super().__init__()
    #         self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
    #         self.ff = nn.Sequential(
    #             nn.Linear(embedding_dim, ff_dim),
    #             nn.ReLU(),
    #             nn.Linear(ff_dim, embedding_dim)
    #         )
    #         self.norm1 = nn.LayerNorm(embedding_dim)
    #         self.norm2 = nn.LayerNorm(embedding_dim)

    #     def forward(self, x):
    #         attn_output, _ = self.attn(x, x, x)
    #         x = self.norm1(x + attn_output)
    #         x = self.norm2(x + self.ff(x))
    #         return x

    # class GlucoseTransformerEncoder(nn.Module):
    #     def __init__(self, input_dim, seq_len, embedding_dim=64, num_heads=8, ff_dim=128, num_layers=4):
    #         super().__init__()

    #         self.embedding = nn.Linear(input_dim, embedding_dim)
    #         self.pos_embedding = nn.Embedding(seq_len, embedding_dim)
    #         self.transformer_blocks = nn.ModuleList([
    #             TransformerBlock(embedding_dim, num_heads, ff_dim)
    #             for _ in range(num_layers)
    #         ])
    #         self.output = nn.Linear(embedding_dim, 1)

    #     def forward(self, x):
    #         pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
    #         x = self.embedding(x) + self.pos_embedding(pos_ids)
    #         for block in self.transformer_blocks:
    #             x = block(x)
    #         x = x.mean(dim=1)
    #         return self.output(x)

class GlucoseTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, embedding_dim=64, num_heads=8, ff_dim=128, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # layers
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.output_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size = x.size(0)
        device = x.device
        
        # positional encoding
        pos_ids = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # embed
        x = self.embedding(x) + self.pos_embedding(pos_ids)
        
        # attention block
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # global average pooling over sequence length
        x = x.mean(dim=1)
        
        return self.output_head(x)